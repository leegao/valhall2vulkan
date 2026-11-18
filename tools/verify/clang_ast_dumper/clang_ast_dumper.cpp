#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace cl = llvm::cl;

static cl::OptionCategory DumperCategory("clang-ast-dumper options");
static cl::list<std::string> InputFiles(cl::Positional, cl::desc("<source files>"), cl::OneOrMore,
                                        cl::cat(DumperCategory));
static cl::opt<std::string> OutputPath("o", cl::desc("Output JSON path"), cl::value_desc("file"),
                                       cl::Required, cl::cat(DumperCategory));
static cl::opt<std::string> RepoRoot("root", cl::desc("Repository root for relative paths"),
                                     cl::init(""), cl::cat(DumperCategory));
static cl::opt<bool> Verbose("verbose", cl::desc("Print diagnostic and timing summary to stderr"),
                             cl::init(false), cl::cat(DumperCategory));
static cl::opt<bool> Strict("strict", cl::desc("Fail if any unhandled AST nodes are encountered"),
                            cl::init(false), cl::cat(DumperCategory));

namespace {

std::string makeRelativePath(llvm::StringRef path) {
    if (RepoRoot.empty()) return path.str();
    llvm::SmallString<256> absPath(path);
    llvm::sys::fs::make_absolute(absPath);
    llvm::SmallString<256> absRoot(RepoRoot);
    llvm::sys::fs::make_absolute(absRoot);
    std::string p = absPath.str().str();
    std::string r = absRoot.str().str();
    if (!r.empty() && r.back() != '/') r.push_back('/');
    if (p.rfind(r, 0) == 0) {
        return p.substr(r.size());
    }
    return path.str();
}

class JsonAstSerializer {
public:
    explicit JsonAstSerializer(clang::ASTContext& ctx)
        : ctx_(ctx), sm_(ctx.getSourceManager()), policy_(ctx.getLangOpts()) {
        policy_.SuppressTagKeyword = true;
        policy_.Bool = true;
        policy_.SuppressScope = false;
    }

    std::string formatType(clang::QualType qt) const {
        if (qt.isNull()) return "";
        return qt.getAsString(policy_);
    }

    std::string formatCanonicalType(clang::QualType qt) const {
        if (qt.isNull()) return "";
        return qt.getCanonicalType().getAsString(policy_);
    }

    llvm::json::Object makeLoc(clang::SourceLocation loc) const {
        llvm::json::Object obj;
        if (loc.isInvalid()) return obj;
        clang::SourceLocation spelling = sm_.getSpellingLoc(loc);
        obj["line"] = static_cast<int64_t>(sm_.getSpellingLineNumber(spelling));
        obj["col"] = static_cast<int64_t>(sm_.getSpellingColumnNumber(spelling));
        return obj;
    }

    const clang::Expr* unwrapExpr(const clang::Expr* e) const {
        while (e) {
            if (auto* p = llvm::dyn_cast<clang::ParenExpr>(e)) {
                e = p->getSubExpr();
            } else if (auto* c = llvm::dyn_cast<clang::ExprWithCleanups>(e)) {
                e = c->getSubExpr();
            } else if (auto* m = llvm::dyn_cast<clang::MaterializeTemporaryExpr>(e)) {
                e = m->getSubExpr();
            } else if (auto* b = llvm::dyn_cast<clang::CXXBindTemporaryExpr>(e)) {
                e = b->getSubExpr();
            } else if (auto* s = llvm::dyn_cast<clang::SubstNonTypeTemplateParmExpr>(e)) {
                e = s->getReplacement();
            } else if (auto* ce = llvm::dyn_cast<clang::ConstantExpr>(e)) {
                e = ce->getSubExpr();
            } else if (auto* sil = llvm::dyn_cast<clang::CXXStdInitializerListExpr>(e)) {
                e = sil->getSubExpr();
            } else if (auto* da = llvm::dyn_cast<clang::CXXDefaultArgExpr>(e)) {
                e = da->getExpr();
            } else if (auto* ic = llvm::dyn_cast<clang::ImplicitCastExpr>(e)) {
                // Keep implicit casts only if they are meaningful semantic conversions?
                // Actually, since every expression carries its resolved `type` and
                // `canonical_type`, stripping ImplicitCastExpr keeps the AST clean.
                e = ic->getSubExpr();
            } else {
                break;
            }
        }
        return e;
    }

    llvm::json::Object serializeFunction(const clang::FunctionDecl* fd) {
        llvm::json::Object obj;
        obj["name"] = fd->getNameAsString();
        obj["qualified_name"] = fd->getQualifiedNameAsString();

        // Compute namespace / class scope
        std::string fullNs;
        std::string shortNs;
        if (auto* md = llvm::dyn_cast<clang::CXXMethodDecl>(fd)) {
            shortNs = md->getParent()->getNameAsString();
            fullNs = md->getParent()->getQualifiedNameAsString();
        } else if (const clang::DeclContext* dc = fd->getDeclContext()) {
            if (auto* nd = llvm::dyn_cast<clang::NamespaceDecl>(dc)) {
                shortNs = nd->getNameAsString();
                fullNs = nd->getQualifiedNameAsString();
            }
        }
        obj["namespace"] = fullNs;
        obj["short_namespace"] = (shortNs == "compiler" ? "" : shortNs);

        obj["ret_type"] = formatType(fd->getReturnType());
        obj["canonical_ret_type"] = formatCanonicalType(fd->getReturnType());

        llvm::json::Array params;
        for (const clang::ParmVarDecl* p : fd->parameters()) {
            llvm::json::Object po;
            po["name"] = p->getNameAsString();
            po["type"] = formatType(p->getType());
            po["canonical_type"] = formatCanonicalType(p->getType());
            params.push_back(std::move(po));
        }
        obj["params"] = std::move(params);

        llvm::json::Array bodyArr;
        if (fd->hasBody()) {
            serializeStmtAsBlockItems(fd->getBody(), bodyArr);
        }
        obj["body"] = std::move(bodyArr);

        clang::SourceLocation startLoc = sm_.getSpellingLoc(fd->getBeginLoc());
        clang::SourceLocation endLoc = sm_.getSpellingLoc(fd->getEndLoc());
        obj["source_file"] = makeRelativePath(sm_.getFilename(startLoc));
        obj["start_line"] = static_cast<int64_t>(sm_.getSpellingLineNumber(startLoc));
        obj["end_line"] = static_cast<int64_t>(sm_.getSpellingLineNumber(endLoc));
        obj["is_static"] = (fd->getStorageClass() == clang::SC_Static);
        obj["is_inline"] = fd->isInlined();
        obj["is_method"] = llvm::isa<clang::CXXMethodDecl>(fd);
        return obj;
    }

    void serializeStmtAsBlockItems(const clang::Stmt* s, llvm::json::Array& out) {
        if (!s) return;
        if (auto* cs = llvm::dyn_cast<clang::CompoundStmt>(s)) {
            for (const clang::Stmt* child : cs->body()) {
                llvm::json::Value v = serializeStmt(child);
                if (!v.getAsNull()) out.push_back(std::move(v));
            }
        } else {
            llvm::json::Value v = serializeStmt(s);
            if (!v.getAsNull()) out.push_back(std::move(v));
        }
    }

    llvm::json::Object serializeVarDecl(const clang::VarDecl* vd) {
        llvm::json::Object obj;
        obj["kind"] = "var_decl";
        obj["name"] = vd->getNameAsString();
        obj["type"] = formatType(vd->getType());
        obj["canonical_type"] = formatCanonicalType(vd->getType());
        obj["loc"] = makeLoc(vd->getLocation());

        bool isStructZero = false;
        if (const clang::Expr* init = vd->getInit()) {
            const clang::Expr* raw = unwrapExpr(init);
            if (auto* il = llvm::dyn_cast_or_null<clang::InitListExpr>(raw)) {
                bool allDefaultOrZero = true;
                for (const clang::Expr* elem : il->inits()) {
                    if (!llvm::isa<clang::CXXDefaultInitExpr>(elem) &&
                        !llvm::isa<clang::ImplicitValueInitExpr>(elem)) {
                        allDefaultOrZero = false;
                        break;
                    }
                }
                if (il->getNumInits() == 0 || allDefaultOrZero) {
                    isStructZero = true;
                }
            } else if (auto* ce = llvm::dyn_cast_or_null<clang::CXXConstructExpr>(raw)) {
                if (ce->getNumArgs() == 0 && vd->getInitStyle() == clang::VarDecl::ListInit) {
                    isStructZero = true;
                }
            }
            obj["is_struct_zero"] = isStructZero;
            if (isStructZero) {
                obj["init"] = nullptr;
            } else {
                obj["init"] = serializeExpr(init);
            }
        } else {
            obj["is_struct_zero"] = false;
            obj["init"] = nullptr;
        }
        return obj;
    }

    llvm::json::Value serializeStmt(const clang::Stmt* s) {
        if (!s) return nullptr;
        if (llvm::isa<clang::NullStmt>(s)) return nullptr;

        if (auto* cs = llvm::dyn_cast<clang::CompoundStmt>(s)) {
            llvm::json::Object obj;
            obj["kind"] = "block";
            obj["loc"] = makeLoc(cs->getBeginLoc());
            llvm::json::Array stmts;
            for (const clang::Stmt* child : cs->body()) {
                llvm::json::Value v = serializeStmt(child);
                if (!v.getAsNull()) stmts.push_back(std::move(v));
            }
            obj["stmts"] = std::move(stmts);
            return obj;
        }

        if (auto* ds = llvm::dyn_cast<clang::DeclStmt>(s)) {
            llvm::json::Array decls;
            for (const clang::Decl* d : ds->decls()) {
                if (auto* vd = llvm::dyn_cast<clang::VarDecl>(d)) {
                    decls.push_back(serializeVarDecl(vd));
                }
            }
            if (decls.empty()) return nullptr;
            if (decls.size() == 1) return std::move(decls[0]);
            llvm::json::Object obj;
            obj["kind"] = "multi_var_decl";
            obj["loc"] = makeLoc(ds->getBeginLoc());
            obj["decls"] = std::move(decls);
            return obj;
        }

        if (auto* rs = llvm::dyn_cast<clang::ReturnStmt>(s)) {
            llvm::json::Object obj;
            obj["kind"] = "return";
            obj["loc"] = makeLoc(rs->getReturnLoc());
            obj["expr"] = rs->getRetValue() ? serializeExpr(rs->getRetValue()) : nullptr;
            return obj;
        }

        if (auto* ifs = llvm::dyn_cast<clang::IfStmt>(s)) {
            llvm::json::Object obj;
            obj["kind"] = "if";
            obj["loc"] = makeLoc(ifs->getIfLoc());
            obj["init"] = ifs->getInit() ? serializeStmt(ifs->getInit()) : nullptr;
            obj["cond"] = serializeExpr(ifs->getCond());
            llvm::json::Array thenBody;
            serializeStmtAsBlockItems(ifs->getThen(), thenBody);
            obj["then_body"] = std::move(thenBody);
            if (ifs->getElse()) {
                llvm::json::Array elseBody;
                serializeStmtAsBlockItems(ifs->getElse(), elseBody);
                obj["else_body"] = std::move(elseBody);
            } else {
                obj["else_body"] = nullptr;
            }
            return obj;
        }

        if (auto* fs = llvm::dyn_cast<clang::ForStmt>(s)) {
            llvm::json::Object obj;
            obj["kind"] = "for";
            obj["loc"] = makeLoc(fs->getForLoc());
            obj["init"] = fs->getInit() ? serializeStmt(fs->getInit()) : nullptr;
            obj["cond"] = fs->getCond() ? serializeExpr(fs->getCond()) : nullptr;
            obj["step"] = fs->getInc() ? serializeExpr(fs->getInc()) : nullptr;
            llvm::json::Array body;
            serializeStmtAsBlockItems(fs->getBody(), body);
            obj["body"] = std::move(body);
            return obj;
        }

        if (auto* rfs = llvm::dyn_cast<clang::CXXForRangeStmt>(s)) {
            llvm::json::Object obj;
            obj["kind"] = "range_for";
            obj["loc"] = makeLoc(rfs->getForLoc());
            if (const clang::VarDecl* loopVar = rfs->getLoopVariable()) {
                obj["var"] = serializeVarDecl(loopVar);
            } else {
                obj["var"] = nullptr;
            }
            obj["range"] = serializeExpr(rfs->getRangeInit());
            llvm::json::Array body;
            serializeStmtAsBlockItems(rfs->getBody(), body);
            obj["body"] = std::move(body);
            return obj;
        }

        if (auto* ws = llvm::dyn_cast<clang::WhileStmt>(s)) {
            llvm::json::Object obj;
            obj["kind"] = "while";
            obj["loc"] = makeLoc(ws->getWhileLoc());
            obj["cond"] = serializeExpr(ws->getCond());
            llvm::json::Array body;
            serializeStmtAsBlockItems(ws->getBody(), body);
            obj["body"] = std::move(body);
            return obj;
        }

        if (auto* dws = llvm::dyn_cast<clang::DoStmt>(s)) {
            llvm::json::Object obj;
            obj["kind"] = "do_while";
            obj["loc"] = makeLoc(dws->getDoLoc());
            obj["cond"] = serializeExpr(dws->getCond());
            llvm::json::Array body;
            serializeStmtAsBlockItems(dws->getBody(), body);
            obj["body"] = std::move(body);
            return obj;
        }

        if (auto* sw = llvm::dyn_cast<clang::SwitchStmt>(s)) {
            llvm::json::Object obj;
            obj["kind"] = "switch";
            obj["loc"] = makeLoc(sw->getSwitchLoc());
            obj["expr"] = serializeExpr(sw->getCond());
            llvm::json::Array cases;
            serializeSwitchBody(sw->getBody(), cases);
            obj["cases"] = std::move(cases);
            return obj;
        }

        if (llvm::isa<clang::BreakStmt>(s)) {
            llvm::json::Object obj;
            obj["kind"] = "break";
            obj["loc"] = makeLoc(s->getBeginLoc());
            return obj;
        }

        if (llvm::isa<clang::ContinueStmt>(s)) {
            llvm::json::Object obj;
            obj["kind"] = "continue";
            obj["loc"] = makeLoc(s->getBeginLoc());
            return obj;
        }

        if (auto* e = llvm::dyn_cast<clang::Expr>(s)) {
            llvm::json::Object obj;
            obj["kind"] = "expr_stmt";
            obj["loc"] = makeLoc(e->getBeginLoc());
            obj["expr"] = serializeExpr(e);
            return obj;
        }

        llvm::json::Object fallback;
        unhandledCount_++;
        fallback["kind"] = "unhandled_stmt";
        fallback["stmt_class"] = s->getStmtClassName();
        fallback["loc"] = makeLoc(s->getBeginLoc());
        return fallback;
    }

    void serializeSwitchBody(const clang::Stmt* body, llvm::json::Array& cases) {
        if (!body) return;
        std::vector<const clang::Stmt*> flat;
        if (auto* cs = llvm::dyn_cast<clang::CompoundStmt>(body)) {
            for (const clang::Stmt* st : cs->body()) {
                flat.push_back(st);
            }
        } else {
            flat.push_back(body);
        }

        llvm::json::Object* curCase = nullptr;
        auto pushNewCase = [&](llvm::json::Value val, clang::SourceLocation loc) {
            llvm::json::Object c;
            c["value"] = std::move(val);
            c["loc"] = makeLoc(loc);
            c["body"] = llvm::json::Array();
            cases.push_back(std::move(c));
            curCase = cases.back().getAsObject();
        };

        std::function<void(const clang::Stmt*)> processStmt = [&](const clang::Stmt* st) {
            if (!st) return;
            if (auto* cs = llvm::dyn_cast<clang::CaseStmt>(st)) {
                pushNewCase(serializeExpr(cs->getLHS()), cs->getCaseLoc());
                processStmt(cs->getSubStmt());
            } else if (auto* ds = llvm::dyn_cast<clang::DefaultStmt>(st)) {
                pushNewCase(nullptr, ds->getDefaultLoc());
                processStmt(ds->getSubStmt());
            } else {
                if (!curCase) {
                    pushNewCase(nullptr, st->getBeginLoc());
                }
                llvm::json::Value v = serializeStmt(st);
                if (!v.getAsNull()) {
                    curCase->getArray("body")->push_back(std::move(v));
                }
            }
        };

        for (const clang::Stmt* st : flat) {
            processStmt(st);
        }
    }

    llvm::json::Value serializeExpr(const clang::Expr* rawExpr) {
        const clang::Expr* e = unwrapExpr(rawExpr);
        if (!e) return nullptr;

        llvm::json::Object obj;
        obj["loc"] = makeLoc(e->getBeginLoc());
        obj["type"] = formatType(e->getType());
        obj["canonical_type"] = formatCanonicalType(e->getType());

        if (auto* bo = llvm::dyn_cast<clang::BinaryOperator>(e)) {
            if (bo->isAssignmentOp()) {
                obj["kind"] = "assign";
                obj["op"] = bo->getOpcodeStr().str();
                obj["target"] = serializeExpr(bo->getLHS());
                obj["value"] = serializeExpr(bo->getRHS());
            } else {
                obj["kind"] = "binary";
                obj["op"] = bo->getOpcodeStr().str();
                obj["left"] = serializeExpr(bo->getLHS());
                obj["right"] = serializeExpr(bo->getRHS());
            }
            return obj;
        }

        if (auto* uo = llvm::dyn_cast<clang::UnaryOperator>(e)) {
            obj["kind"] = "unary";
            obj["op"] = clang::UnaryOperator::getOpcodeStr(uo->getOpcode()).str();
            obj["operand"] = serializeExpr(uo->getSubExpr());
            obj["postfix"] = uo->isPostfix();
            return obj;
        }

        if (auto* co = llvm::dyn_cast<clang::ConditionalOperator>(e)) {
            obj["kind"] = "ternary";
            obj["cond"] = serializeExpr(co->getCond());
            obj["then_expr"] = serializeExpr(co->getTrueExpr());
            obj["else_expr"] = serializeExpr(co->getFalseExpr());
            return obj;
        }

        if (auto* ec = llvm::dyn_cast<clang::ExplicitCastExpr>(e)) {
            obj["kind"] = "cast";
            obj["cast_kind"] = ec->getStmtClassName();
            obj["target_type"] = formatType(ec->getTypeAsWritten());
            obj["expr"] = serializeExpr(ec->getSubExpr());
            return obj;
        }

        if (auto* opCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(e)) {
            clang::OverloadedOperatorKind ok = opCall->getOperator();
            std::string opStr = clang::getOperatorSpelling(ok);
            if (ok == clang::OO_Subscript && opCall->getNumArgs() == 2) {
                obj["kind"] = "index";
                obj["obj"] = serializeExpr(opCall->getArg(0));
                obj["index"] = serializeExpr(opCall->getArg(1));
                return obj;
            }
            if ((ok == clang::OO_Equal || clang::isCompoundAssignmentOperator(ok)) && opCall->getNumArgs() == 2) {
                obj["kind"] = "assign";
                obj["op"] = opStr;
                obj["target"] = serializeExpr(opCall->getArg(0));
                obj["value"] = serializeExpr(opCall->getArg(1));
                return obj;
            }
            if (opCall->getNumArgs() == 2 && ok != clang::OO_Call) {
                obj["kind"] = "binary";
                obj["op"] = opStr;
                obj["left"] = serializeExpr(opCall->getArg(0));
                obj["right"] = serializeExpr(opCall->getArg(1));
                return obj;
            }
            if (opCall->getNumArgs() == 1 && ok != clang::OO_Call) {
                obj["kind"] = "unary";
                obj["op"] = opStr;
                obj["operand"] = serializeExpr(opCall->getArg(0));
                obj["postfix"] = false;
                return obj;
            }
        }

        if (auto* mce = llvm::dyn_cast<clang::CXXMemberCallExpr>(e)) {
            obj["kind"] = "call";
            const clang::Expr* callee = unwrapExpr(mce->getCallee());
            if (auto* me = llvm::dyn_cast_or_null<clang::MemberExpr>(callee)) {
                llvm::json::Object mem;
                mem["kind"] = "member";
                mem["loc"] = makeLoc(me->getBeginLoc());
                mem["obj"] = serializeExpr(me->getBase());
                mem["member"] = me->getMemberDecl()->getNameAsString();
                mem["is_arrow"] = me->isArrow();
                obj["callee"] = std::move(mem);
            } else {
                obj["callee"] = serializeExpr(callee);
            }
            if (const clang::FunctionDecl* fd = mce->getDirectCallee()) {
                obj["qualified_callee"] = fd->getQualifiedNameAsString();
            } else {
                obj["qualified_callee"] = "";
            }
            llvm::json::Array args;
            for (const clang::Expr* arg : mce->arguments()) {
                if (!llvm::isa<clang::CXXDefaultArgExpr>(arg)) {
                    args.push_back(serializeExpr(arg));
                }
            }
            obj["args"] = std::move(args);
            obj["template_args"] = llvm::json::Array();
            return obj;
        }

        if (auto* ce = llvm::dyn_cast<clang::CallExpr>(e)) {
            obj["kind"] = "call";
            obj["callee"] = serializeExpr(ce->getCallee());
            std::string qualCallee;
            llvm::json::Array tplArgs;
            if (const clang::FunctionDecl* fd = ce->getDirectCallee()) {
                qualCallee = fd->getQualifiedNameAsString();
                if (const clang::TemplateArgumentList* tal = fd->getTemplateSpecializationArgs()) {
                    for (unsigned i = 0; i < tal->size(); ++i) {
                        const clang::TemplateArgument& ta = tal->get(i);
                        if (ta.getKind() == clang::TemplateArgument::Type) {
                            tplArgs.push_back(formatType(ta.getAsType()));
                        }
                    }
                }
            }
            obj["qualified_callee"] = qualCallee;
            llvm::json::Array args;
            for (const clang::Expr* arg : ce->arguments()) {
                if (!llvm::isa<clang::CXXDefaultArgExpr>(arg)) {
                    args.push_back(serializeExpr(arg));
                }
            }
            obj["args"] = std::move(args);
            obj["template_args"] = std::move(tplArgs);
            return obj;
        }

        if (auto* me = llvm::dyn_cast<clang::MemberExpr>(e)) {
            // Implicit `this->member` inside Pass1Context methods
            if (me->isImplicitAccess()) {
                obj["kind"] = "ident";
                obj["name"] = me->getMemberDecl()->getNameAsString();
                obj["implicit_this"] = true;
                return obj;
            }
            obj["kind"] = "member";
            obj["obj"] = serializeExpr(me->getBase());
            obj["member"] = me->getMemberDecl()->getNameAsString();
            obj["is_arrow"] = me->isArrow();
            return obj;
        }

        if (auto* ase = llvm::dyn_cast<clang::ArraySubscriptExpr>(e)) {
            obj["kind"] = "index";
            obj["obj"] = serializeExpr(ase->getBase());
            obj["index"] = serializeExpr(ase->getIdx());
            return obj;
        }

        if (auto* dre = llvm::dyn_cast<clang::DeclRefExpr>(e)) {
            const clang::NamedDecl* nd = dre->getFoundDecl();
            if (dre->hasQualifier() || llvm::isa<clang::EnumConstantDecl>(nd)) {
                std::string qname = nd->getQualifiedNameAsString();
                llvm::json::Array parts;
                size_t pos = 0;
                while (pos < qname.size()) {
                    size_t next = qname.find("::", pos);
                    if (next == std::string::npos) {
                        parts.push_back(qname.substr(pos));
                        break;
                    }
                    parts.push_back(qname.substr(pos, next - pos));
                    pos = next + 2;
                }
                if (parts.size() > 1) {
                    obj["kind"] = "scoped_ident";
                    obj["parts"] = std::move(parts);
                    obj["qualified_name"] = qname;
                    return obj;
                }
            }
            obj["kind"] = "ident";
            obj["name"] = nd->getNameAsString();
            return obj;
        }

        if (auto* il = llvm::dyn_cast<clang::IntegerLiteral>(e)) {
            obj["kind"] = "literal";
            obj["lit_kind"] = "int";
            llvm::SmallString<32> s;
            bool isSigned = il->getType()->isSignedIntegerType();
            il->getValue().toString(s, 10, isSigned);
            obj["value"] = s.str().str();
            if (il->getValue().getActiveBits() <= 53) {
                if (isSigned) obj["int_value"] = il->getValue().getSExtValue();
                else obj["int_value"] = static_cast<int64_t>(il->getValue().getZExtValue());
            }
            return obj;
        }

        if (auto* fl = llvm::dyn_cast<clang::FloatingLiteral>(e)) {
            obj["kind"] = "literal";
            obj["lit_kind"] = "float";
            llvm::SmallString<32> s;
            fl->getValue().toString(s);
            obj["value"] = s.str().str();
            return obj;
        }

        if (auto* bl = llvm::dyn_cast<clang::CXXBoolLiteralExpr>(e)) {
            obj["kind"] = "literal";
            obj["lit_kind"] = "bool";
            obj["value"] = bl->getValue();
            return obj;
        }

        if (auto* sl = llvm::dyn_cast<clang::StringLiteral>(e)) {
            obj["kind"] = "literal";
            obj["lit_kind"] = "string";
            obj["value"] = sl->getString().str();
            return obj;
        }

        if (auto* clit = llvm::dyn_cast<clang::CharacterLiteral>(e)) {
            obj["kind"] = "literal";
            obj["lit_kind"] = "char";
            obj["value"] = std::string(1, static_cast<char>(clit->getValue()));
            return obj;
        }

        if (llvm::isa<clang::CXXNullPtrLiteralExpr>(e)) {
            obj["kind"] = "literal";
            obj["lit_kind"] = "null";
            obj["value"] = nullptr;
            return obj;
        }

        if (auto* ile = llvm::dyn_cast<clang::InitListExpr>(e)) {
            obj["kind"] = "init_list";
            llvm::json::Array items;
            for (const clang::Expr* item : ile->inits()) {
                items.push_back(serializeExpr(item));
            }
            obj["items"] = std::move(items);
            return obj;
        }

        if (auto* cce = llvm::dyn_cast<clang::CXXConstructExpr>(e)) {
            // If constructing from a single argument (e.g. std::string("a") or SsaNode copy),
            // unwrap single-arg implicit/conversion constructors, otherwise emit init_list/call.
            if (cce->getNumArgs() == 1 && !cce->isListInitialization()) {
                return serializeExpr(cce->getArg(0));
            }
            obj["kind"] = "init_list";
            llvm::json::Array items;
            for (const clang::Expr* arg : cce->arguments()) {
                if (!llvm::isa<clang::CXXDefaultArgExpr>(arg)) {
                    items.push_back(serializeExpr(arg));
                }
            }
            obj["items"] = std::move(items);
            return obj;
        }

        if (auto* le = llvm::dyn_cast<clang::LambdaExpr>(e)) {
            obj["kind"] = "lambda";
            const clang::CXXMethodDecl* callOp = le->getCallOperator();
            llvm::json::Array params;
            if (callOp) {
                for (const clang::ParmVarDecl* p : callOp->parameters()) {
                    llvm::json::Object po;
                    po["name"] = p->getNameAsString();
                    po["type"] = formatType(p->getType());
                    params.push_back(std::move(po));
                }
                obj["ret_type"] = formatType(callOp->getReturnType());
            }
            obj["params"] = std::move(params);
            llvm::json::Array body;
            serializeStmtAsBlockItems(le->getBody(), body);
            obj["body"] = std::move(body);
            return obj;
        }

        if (auto* uett = llvm::dyn_cast<clang::UnaryExprOrTypeTraitExpr>(e)) {
            obj["kind"] = "sizeof";
            if (uett->isArgumentType()) {
                obj["target_type"] = formatType(uett->getArgumentType());
            } else {
                obj["expr"] = serializeExpr(uett->getArgumentExpr());
            }
            return obj;
        }

        if (llvm::isa<clang::CXXThisExpr>(e)) {
            obj["kind"] = "ident";
            obj["name"] = "this";
            return obj;
        }

        if (llvm::isa<clang::ImplicitValueInitExpr>(e)) {
            obj["kind"] = "zero_init";
            return obj;
        }
        if (auto* die = llvm::dyn_cast<clang::CXXDefaultInitExpr>(e)) {
            if (const clang::Expr* sub = die->getExpr()) {
                return serializeExpr(sub);
            }
            obj["kind"] = "default_init";
            return obj;
        }

        unhandledCount_++;
        obj["kind"] = "unhandled_expr";
        obj["expr_class"] = e->getStmtClassName();
        return obj;
    }

private:
    clang::ASTContext& ctx_;
    clang::SourceManager& sm_;
    clang::PrintingPolicy policy_;
public:
    struct VerifierZ3Attr {
        std::string receiver;
        std::string expr;
    };

    static bool parseCppStringLiteral(const char*& cur, const char* end, std::string& out) {
        while (cur < end && (*cur == ' ' || *cur == '\t' || *cur == '\n' || *cur == '\r')) ++cur;
        if (cur >= end || *cur != '"') return false;
        ++cur;
        out.clear();
        while (cur < end) {
            char c = *cur++;
            if (c == '"') return true;
            if (c == '\\' && cur < end) {
                char esc = *cur++;
                if (esc == 'n') out.push_back('\n');
                else if (esc == 't') out.push_back('\t');
                else if (esc == 'r') out.push_back('\r');
                else out.push_back(esc);
            } else {
                out.push_back(c);
            }
        }
        return false;
    }

    bool extractVerifierZ3Attr(const clang::FunctionDecl* fd, VerifierZ3Attr& out) const {
        if (!fd) return false;
        clang::SourceLocation loc = sm_.getSpellingLoc(fd->getLocation());
        if (!loc.isValid()) return false;
        clang::FileID fid = sm_.getFileID(loc);
        bool invalid = false;
        llvm::StringRef buf = sm_.getBufferData(fid, &invalid);
        if (invalid) return false;
        unsigned endOff = sm_.getFileOffset(loc);
        if (endOff > buf.size()) return false;
        unsigned startOff = (endOff > 1024) ? (endOff - 1024) : 0;
        // Stop at previous declaration/scope boundary (; { } :) outside quotes
        bool inStr = false;
        for (unsigned i = endOff; i > startOff; --i) {
            char c = buf[i - 1];
            if (c == '"' && (i < 2 || buf[i - 2] != '\\')) inStr = !inStr;
            if (!inStr && (c == ';' || c == '{' || c == '}')) {
                startOff = i;
                break;
            }
        }
        llvm::StringRef window = buf.slice(startOff, endOff);
        size_t pos = window.find("[[");
        while (pos != llvm::StringRef::npos) {
            const char* cur = window.data() + pos + 2;
            const char* wend = window.data() + window.size();
            auto skipWs = [&]() {
                while (cur < wend && (*cur == ' ' || *cur == '\t' || *cur == '\n' || *cur == '\r')) ++cur;
            };
            skipWs();
            llvm::StringRef rest(cur, static_cast<size_t>(wend - cur));
            if (rest.starts_with("verifier::z3")) {
                cur += 12;
                skipWs();
                if (cur < wend && *cur == '(') {
                    ++cur;
                    std::string recv, expr;
                    if (parseCppStringLiteral(cur, wend, recv)) {
                        skipWs();
                        if (cur < wend && *cur == ',') {
                            ++cur;
                            if (parseCppStringLiteral(cur, wend, expr)) {
                                out.receiver = std::move(recv);
                                out.expr = std::move(expr);
                                return true;
                            }
                        }
                    }
                }
            }
            pos = window.find("[[", pos + 2);
        }
        return false;
    }

private:
    size_t unhandledCount_ = 0;
public:
    size_t unhandledCount() const { return unhandledCount_; }
};

class DumperVisitor : public clang::RecursiveASTVisitor<DumperVisitor> {
public:
    DumperVisitor(clang::ASTContext& ctx, llvm::json::Array& functions,
                  llvm::json::Object& verifierAttrs, size_t& unhandledCount)
        : ctx_(ctx), sm_(ctx.getSourceManager()), serializer_(ctx),
          functions_(functions), verifierAttrs_(verifierAttrs), unhandledCount_(unhandledCount) {}

    bool TraverseDecl(clang::Decl* d) {
        if (!d) return true;
        if (llvm::isa<clang::TranslationUnitDecl>(d) ||
            llvm::isa<clang::NamespaceDecl>(d) ||
            llvm::isa<clang::LinkageSpecDecl>(d)) {
            return clang::RecursiveASTVisitor<DumperVisitor>::TraverseDecl(d);
        }
        if (d->getLocation().isValid() && sm_.isInSystemHeader(d->getLocation())) {
            return true;
        }
        if (d->getLocation().isValid() && !sm_.isInMainFile(d->getLocation())) {
            if (llvm::isa<clang::CXXRecordDecl>(d) ||
                llvm::isa<clang::FunctionDecl>(d) ||
                llvm::isa<clang::CXXMethodDecl>(d)) {
                return clang::RecursiveASTVisitor<DumperVisitor>::TraverseDecl(d);
            }
            return true;
        }
        return clang::RecursiveASTVisitor<DumperVisitor>::TraverseDecl(d);
    }

    bool TraverseFunctionDecl(clang::FunctionDecl* fd) {
        if (!fd) return true;
        if (!fd->isImplicit() && !fd->isDependentContext() &&
            fd->getLocation().isValid() && !sm_.isInSystemHeader(fd->getLocation())) {
            JsonAstSerializer::VerifierZ3Attr vattr;
            if (serializer_.extractVerifierZ3Attr(fd, vattr)) {
                llvm::json::Object ao;
                ao["receiver"] = vattr.receiver;
                ao["expr"] = vattr.expr;
                verifierAttrs_[fd->getQualifiedNameAsString()] = std::move(ao);
            }
        }
        if (fd->hasBody() && fd->isThisDeclarationADefinition() &&
            !fd->isImplicit() && !fd->isDependentContext() &&
            sm_.isInMainFile(fd->getLocation())) {
            functions_.push_back(serializer_.serializeFunction(fd));
            unhandledCount_ = serializer_.unhandledCount();
        }
        // Do not descend into function bodies via RecursiveASTVisitor;
        // JsonAstSerializer already walks the body of main-file functions.
        return true;
    }

    bool TraverseCXXMethodDecl(clang::CXXMethodDecl* md) {
        return TraverseFunctionDecl(md);
    }

private:
    clang::ASTContext& ctx_;
    clang::SourceManager& sm_;
    JsonAstSerializer serializer_;
    llvm::json::Array& functions_;
    llvm::json::Object& verifierAttrs_;
    size_t& unhandledCount_;
    bool verbose_ = false;
};

class DumperConsumer : public clang::ASTConsumer {
public:
    DumperConsumer(llvm::json::Array& functions, llvm::json::Object& verifierAttrs,
                   size_t& unhandledCount, bool verbose)
        : functions_(functions), verifierAttrs_(verifierAttrs),
          unhandledCount_(unhandledCount), verbose_(verbose) {}

    void HandleTranslationUnit(clang::ASTContext& ctx) override {
        auto tStart = std::chrono::steady_clock::now();
        if (verbose_) {
            llvm::errs() << "[clang_ast_dumper] ParseAST finished, traversing main-file AST...\n";
        }
        DumperVisitor visitor(ctx, functions_, verifierAttrs_, unhandledCount_);
        visitor.TraverseDecl(ctx.getTranslationUnitDecl());
        auto tEnd = std::chrono::steady_clock::now();
        if (verbose_) {
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count();
            llvm::errs() << "[clang_ast_dumper] AST traversal + JSON build: "
                         << functions_.size() << " functions in " << ms << " ms\n";
        }
    }

private:
    llvm::json::Array& functions_;
    llvm::json::Object& verifierAttrs_;
    size_t& unhandledCount_;
    bool verbose_ = false;
};

class DumperAction : public clang::ASTFrontendAction {
public:
    DumperAction(llvm::json::Array& functions, llvm::json::Object& verifierAttrs, size_t& unhandledCount, bool verbose)
        : functions_(functions), verifierAttrs_(verifierAttrs), unhandledCount_(unhandledCount), verbose_(verbose) {}

    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance&,
                                                          llvm::StringRef) override {
        return std::make_unique<DumperConsumer>(functions_, verifierAttrs_, unhandledCount_, verbose_);
    }

private:
    llvm::json::Array& functions_;
    llvm::json::Object& verifierAttrs_;
    size_t& unhandledCount_;
    bool verbose_ = false;
};

class DumperActionFactory : public clang::tooling::FrontendActionFactory {
public:
    DumperActionFactory(llvm::json::Array& functions, llvm::json::Object& verifierAttrs, size_t& unhandledCount, bool verbose)
        : functions_(functions), verifierAttrs_(verifierAttrs), unhandledCount_(unhandledCount), verbose_(verbose) {}

    std::unique_ptr<clang::FrontendAction> create() override {
        return std::make_unique<DumperAction>(functions_, verifierAttrs_, unhandledCount_, verbose_);
    }

private:
    llvm::json::Array& functions_;
    llvm::json::Object& verifierAttrs_;
    size_t& unhandledCount_;
    bool verbose_ = false;
};

} // namespace

int main(int argc, const char** argv) {
    auto t0 = std::chrono::steady_clock::now();

    std::string errMsg;
    std::unique_ptr<clang::tooling::FixedCompilationDatabase> compDb =
        clang::tooling::FixedCompilationDatabase::loadFromCommandLine(argc, argv, errMsg);

    cl::HideUnrelatedOptions(DumperCategory);
    cl::ParseCommandLineOptions(argc, argv, "Clang-21 C++ AST to JSON dumper for Valhall handlers\n");

    if (!compDb) {
        compDb = std::make_unique<clang::tooling::FixedCompilationDatabase>(".", std::vector<std::string>{});
    }

    bool isVerbose = Verbose || (std::getenv("VALHALL_AST_DUMPER_VERBOSE") != nullptr);
    if (isVerbose) {
        llvm::errs() << "[clang_ast_dumper] starting ClangTool on " << InputFiles.size() << " file(s)...\n";
    }
    clang::tooling::ClangTool tool(*compDb, InputFiles);
    llvm::json::Array functions;
    llvm::json::Object verifierAttrs;
    size_t unhandledCount = 0;
    DumperActionFactory factory(functions, verifierAttrs, unhandledCount, isVerbose);
    int rc = tool.run(&factory);

    llvm::json::Object root;
    root["schema_version"] = 1;
    root["generator"] = "clang_ast_dumper (LLVM 21)";
    llvm::json::Array srcArr;
    for (const std::string& f : InputFiles) {
        srcArr.push_back(makeRelativePath(f));
    }
    root["source_files"] = std::move(srcArr);
    root["function_count"] = static_cast<int64_t>(functions.size());
    root["unhandled_count"] = static_cast<int64_t>(unhandledCount);
    root["verifier_attrs"] = std::move(verifierAttrs);
    root["functions"] = std::move(functions);

    std::error_code ec;
    llvm::SmallString<256> outParent = llvm::sys::path::parent_path(OutputPath);
    if (!outParent.empty()) {
        llvm::sys::fs::create_directories(outParent);
    }
    llvm::raw_fd_ostream os(OutputPath, ec, llvm::sys::fs::OF_Text);
    if (ec) {
        llvm::errs() << "[clang_ast_dumper] ERROR: cannot open output '" << OutputPath
                     << "': " << ec.message() << "\n";
        return 1;
    }
    os << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(root)));
    os.flush();

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    if (Strict && unhandledCount > 0) {
        llvm::errs() << "[clang_ast_dumper] ERROR: " << unhandledCount << " unhandled AST node(s) in --strict mode\n";
        return 2;
    }
    if (isVerbose || rc != 0) {
        llvm::errs() << "[clang_ast_dumper] parsed " << InputFiles.size() << " file(s) -> "
                     << OutputPath << " (" << ms << " ms, exit=" << rc << ")\n";
    }
    return rc;
}
