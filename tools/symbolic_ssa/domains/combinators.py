"""
Composable Domain Combinators for `SsaDomain` (inspired by `src/dataflow/domains/`).

Provides:
- `TupleDomain` (Product domain): runs `k` independent `SsaDomain`s in lockstep,
  wrapping values in `TupleVal` and delegating every operation component-wise.
- `PointwiseUnionDomain` (Tagged disjoint-sum domain): combines named member
  `SsaDomain`s into a single lattice with `UNION_BOTTOM`, `UNION_TOP`, same-tag
  delegation, and optional cross-domain op handlers.
- `MultiSetDomain` (Bounded finite multiset / powerset domain): lifts any
  `SsaDomain` to bounded multisets (with multiplicity tracking or deduplicated
  sets) up to `max_k` distinct elements before widening to Top.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from tools.symbolic_ssa.domains.base import SsaDomain


@dataclasses.dataclass(frozen=True)
class TupleVal:
    """Product-domain value holding one component per wrapped `SsaDomain`."""

    elems: Tuple[Any, ...]

    def __getitem__(self, idx: int) -> Any:
        return self.elems[idx]

    def __len__(self) -> int:
        return len(self.elems)


class TupleDomain(SsaDomain):
    """
    Variadic pointwise product domain over `(d_0, d_1, ..., d_{k-1})`.

    Every `SsaDomain` operation is delegated component-wise to each wrapped
    domain, returning a `TupleVal` of the corresponding results.
    """

    def __init__(self, *domains: SsaDomain):
        if not domains:
            raise ValueError("TupleDomain requires at least one wrapped SsaDomain")
        self.domains: Tuple[SsaDomain, ...] = tuple(domains)

    def _unpack(self, val: Any) -> Tuple[Any, ...]:
        if isinstance(val, TupleVal) and len(val.elems) == len(self.domains):
            return val.elems
        return tuple(val for _ in self.domains)

    def _map0(self, method: str, *args: Any, **kwargs: Any) -> TupleVal:
        return TupleVal(tuple(getattr(d, method)(*args, **kwargs) for d in self.domains))

    def _map1(self, method: str, x: Any, *extra: Any, **kwargs: Any) -> TupleVal:
        xs = self._unpack(x)
        return TupleVal(tuple(getattr(d, method)(xi, *extra, **kwargs) for d, xi in zip(self.domains, xs)))

    def _map2(self, method: str, a: Any, b: Any, *extra: Any, **kwargs: Any) -> TupleVal:
        as_, bs = self._unpack(a), self._unpack(b)
        return TupleVal(tuple(getattr(d, method)(ai, bi, *extra, **kwargs) for d, ai, bi in zip(self.domains, as_, bs)))

    def _map3(self, method: str, a: Any, b: Any, c: Any, *extra: Any, **kwargs: Any) -> TupleVal:
        as_, bs, cs = self._unpack(a), self._unpack(b), self._unpack(c)
        return TupleVal(tuple(getattr(d, method)(ai, bi, ci, *extra, **kwargs) for d, ai, bi, ci in zip(self.domains, as_, bs, cs)))

    def bottom(self) -> TupleVal: return self._map0("bottom")
    def top(self) -> TupleVal: return self._map0("top")
    def default_value(self) -> TupleVal: return self._map0("default_value")
    def join(self, a: Any, b: Any) -> TupleVal: return self._map2("join", a, b)
    def meet(self, a: Any, b: Any) -> TupleVal: return self._map2("meet", a, b)
    def leq(self, a: Any, b: Any) -> bool:
        return all(d.leq(ai, bi) for d, ai, bi in zip(self.domains, self._unpack(a), self._unpack(b)))

    def const_u32(self, imm: int) -> TupleVal: return self._map0("const_u32", imm)
    def const_u64(self, imm: int) -> TupleVal: return self._map0("const_u64", imm)
    def const_bool(self, val: bool) -> TupleVal: return self._map0("const_bool", val)

    def const_int_val(self, val: Any) -> Optional[int]:
        for d, xi in zip(self.domains, self._unpack(val)):
            c = d.const_int_val(xi)
            if c is not None:
                return c
        return None

    def mask32(self, val: Any) -> TupleVal: return self._map1("mask32", val)
    def mask64(self, val: Any) -> TupleVal: return self._map1("mask64", val)
    def as_u32(self, val: Any) -> TupleVal: return self._map1("as_u32", val)
    def as_s32(self, val: Any) -> TupleVal: return self._map1("as_s32", val)
    def as_u64(self, val: Any) -> TupleVal: return self._map1("as_u64", val)
    def as_s64(self, val: Any) -> TupleVal: return self._map1("as_s64", val)
    def as_f32(self, val: Any) -> TupleVal: return self._map1("as_f32", val)
    def as_bool(self, val: Any) -> TupleVal: return self._map1("as_bool", val)

    def cast_value(self, op: str, val: Any, val_type: str) -> TupleVal:
        xs = self._unpack(val)
        return TupleVal(tuple(d.cast_value(op, xi, val_type) for d, xi in zip(self.domains, xs)))

    def pack64(self, lo: Any, hi: Any) -> TupleVal: return self._map2("pack64", lo, hi)

    def split64(self, val64: Any) -> Tuple[TupleVal, TupleVal]:
        xs = self._unpack(val64)
        pairs = [d.split64(xi) for d, xi in zip(self.domains, xs)]
        return (TupleVal(tuple(p[0] for p in pairs)), TupleVal(tuple(p[1] for p in pairs)))

    def unpack_half2x16(self, u: Any) -> Tuple[TupleVal, TupleVal]:
        xs = self._unpack(u)
        pairs = [d.unpack_half2x16(xi) for d, xi in zip(self.domains, xs)]
        return (TupleVal(tuple(p[0] for p in pairs)), TupleVal(tuple(p[1] for p in pairs)))

    def pack_half2x16(self, v2: Any) -> TupleVal:
        if isinstance(v2, tuple) and len(v2) >= 2:
            v0s, v1s = self._unpack(v2[0]), self._unpack(v2[1])
            return TupleVal(tuple(d.pack_half2x16((a, b)) for d, a, b in zip(self.domains, v0s, v1s)))
        return self._map1("pack_half2x16", v2)

    def int32_unary(self, op: str, x: Any) -> TupleVal:
        xs = self._unpack(x)
        return TupleVal(tuple(d.int32_unary(op, xi) for d, xi in zip(self.domains, xs)))

    def int32_binary(self, op: str, a: Any, b: Any) -> TupleVal:
        as_, bs = self._unpack(a), self._unpack(b)
        return TupleVal(tuple(d.int32_binary(op, ai, bi) for d, ai, bi in zip(self.domains, as_, bs)))

    def int64_unary(self, op: str, x: Any) -> TupleVal:
        xs = self._unpack(x)
        return TupleVal(tuple(d.int64_unary(op, xi) for d, xi in zip(self.domains, xs)))

    def int64_binary(self, op: str, a: Any, b: Any) -> TupleVal:
        as_, bs = self._unpack(a), self._unpack(b)
        return TupleVal(tuple(d.int64_binary(op, ai, bi) for d, ai, bi in zip(self.domains, as_, bs)))

    def icmp(self, op: str, a: Any, b: Any) -> TupleVal:
        as_, bs = self._unpack(a), self._unpack(b)
        return TupleVal(tuple(d.icmp(op, ai, bi) for d, ai, bi in zip(self.domains, as_, bs)))

    def bfe(self, signed: bool, x: Any, off: Any, width: Any) -> TupleVal:
        xs, offs, ws = self._unpack(x), self._unpack(off), self._unpack(width)
        return TupleVal(tuple(d.bfe(signed, xi, oi, wi) for d, xi, oi, wi in zip(self.domains, xs, offs, ws)))

    def bfi(self, base: Any, insert: Any, off: Any, width: Any) -> TupleVal:
        bs, ins, offs, ws = self._unpack(base), self._unpack(insert), self._unpack(off), self._unpack(width)
        return TupleVal(tuple(d.bfi(bi, ii, oi, wi) for d, bi, ii, oi, wi in zip(self.domains, bs, ins, offs, ws)))

    def idot4(self, op: str, a: Any, b: Any) -> TupleVal:
        as_, bs = self._unpack(a), self._unpack(b)
        return TupleVal(tuple(d.idot4(op, ai, bi) for d, ai, bi in zip(self.domains, as_, bs)))

    def logical_and(self, a: Any, b: Any) -> TupleVal: return self._map2("logical_and", a, b)
    def logical_or(self, a: Any, b: Any) -> TupleVal: return self._map2("logical_or", a, b)
    def logical_not(self, a: Any) -> TupleVal: return self._map1("logical_not", a)
    def select(self, cond: Any, t_val: Any, f_val: Any) -> TupleVal: return self._map3("select", cond, t_val, f_val)

    def fp_unary(self, op: str, x: Any) -> TupleVal:
        xs = self._unpack(x)
        return TupleVal(tuple(d.fp_unary(op, xi) for d, xi in zip(self.domains, xs)))

    def fp_binary(self, op: str, a: Any, b: Any) -> TupleVal:
        as_, bs = self._unpack(a), self._unpack(b)
        return TupleVal(tuple(d.fp_binary(op, ai, bi) for d, ai, bi in zip(self.domains, as_, bs)))

    def fp_clamp(self, x: Any, lo: Any, hi: Any) -> TupleVal: return self._map3("fp_clamp", x, lo, hi)
    def fp_fma(self, a: Any, b: Any, c: Any) -> TupleVal: return self._map3("fp_fma", a, b, c)
    def fp_is_nan(self, x: Any) -> TupleVal: return self._map1("fp_is_nan", x)
    def fp_is_inf(self, x: Any) -> TupleVal: return self._map1("fp_is_inf", x)

    def fp_cmp(self, op: str, a: Any, b: Any) -> TupleVal:
        as_, bs = self._unpack(a), self._unpack(b)
        return TupleVal(tuple(d.fp_cmp(op, ai, bi) for d, ai, bi in zip(self.domains, as_, bs)))

    def convert(self, op: str, x: Any) -> TupleVal:
        xs = self._unpack(x)
        return TupleVal(tuple(d.convert(op, xi) for d, xi in zip(self.domains, xs)))

    def hw_uf(self, env: Any, name: str, *args: Any, out_bits: int = 32, fallback_tag: Optional[str] = None) -> TupleVal:
        unpacked = [self._unpack(a) for a in args]
        return TupleVal(tuple(
            d.hw_uf(env, name, *(u[i] for u in unpacked), out_bits=out_bits, fallback_tag=fallback_tag)
            for i, d in enumerate(self.domains)
        ))

    def atomic_compute(self, env: Any, op: str, op_tag: str, old_val: Any, arg: Any, cmp_val: Any) -> TupleVal:
        os_, as_, cs = self._unpack(old_val), self._unpack(arg), self._unpack(cmp_val)
        return TupleVal(tuple(
            d.atomic_compute(env, op, op_tag, oi, ai, ci)
            for d, oi, ai, ci in zip(self.domains, os_, as_, cs)
        ))


@dataclasses.dataclass(frozen=True)
class UnionVal:
    """Tagged disjoint-sum lattice value (`tag` is `"__bottom__"`, `"__top__"`, or a member domain key)."""

    tag: str
    value: Any = None

    @property
    def is_bottom(self) -> bool:
        return self.tag == "__bottom__"

    @property
    def is_top(self) -> bool:
        return self.tag == "__top__"


UNION_BOTTOM = UnionVal("__bottom__", None)
UNION_TOP = UnionVal("__top__", None)

CrossDomainHandler = Callable[[Any, Any], Any]


class PointwiseUnionDomain(SsaDomain):
    """
    Tagged disjoint-sum lattice over named member `SsaDomain`s.

    Mirrors `valhall::dataflow::UnionDomain` (`src/dataflow/domains/union.h`):
    - Same-tag operations delegate directly to the corresponding member domain.
    - Cross-tag binary operations invoke an optional registered
      `cross_ops[(method, tag_a, tag_b)]` handler, or widen to `UNION_TOP`.
    - `default_tag` designates the member domain used to construct scalar
      constants (`const_u32`, `const_u64`, `const_bool`).
    """

    def __init__(
        self,
        members: Mapping[str, SsaDomain],
        default_tag: Optional[str] = None,
        cross_ops: Optional[Dict[Tuple[str, str, str], CrossDomainHandler]] = None,
    ):
        if not members:
            raise ValueError("PointwiseUnionDomain requires at least one member domain")
        self.members: Dict[str, SsaDomain] = dict(members)
        self.default_tag: str = default_tag if default_tag is not None else next(iter(self.members))
        if self.default_tag not in self.members:
            raise KeyError(f"default_tag {self.default_tag!r} not in members")
        self.cross_ops: Dict[Tuple[str, str, str], CrossDomainHandler] = dict(cross_ops or {})

    def register_cross_op(self, method: str, tag_a: str, tag_b: str, handler: CrossDomainHandler) -> None:
        self.cross_ops[(method, tag_a, tag_b)] = handler

    def lift(self, tag: str, value: Any) -> UnionVal:
        """Wrap a member domain value, collapsing member bottom/top to union bottom/top."""
        dom = self.members[tag]
        if isinstance(value, UnionVal):
            return value
        if value == dom.bottom() and dom.bottom() != dom.top():
            return UNION_BOTTOM
        if value == dom.top() and dom.bottom() != dom.top():
            return UNION_TOP
        return UnionVal(tag, value)

    def _as_union(self, val: Any) -> UnionVal:
        if isinstance(val, UnionVal):
            return val
        return UnionVal(self.default_tag, val)

    def bottom(self) -> UnionVal: return UNION_BOTTOM
    def top(self) -> UnionVal: return UNION_TOP
    def default_value(self) -> UnionVal: return UNION_TOP

    def join(self, a: Any, b: Any) -> UnionVal:
        ua, ub = self._as_union(a), self._as_union(b)
        if ua.is_bottom: return ub
        if ub.is_bottom: return ua
        if ua.is_top or ub.is_top or ua.tag != ub.tag:
            return UNION_TOP
        return self.lift(ua.tag, self.members[ua.tag].join(ua.value, ub.value))

    def meet(self, a: Any, b: Any) -> UnionVal:
        ua, ub = self._as_union(a), self._as_union(b)
        if ua.is_top: return ub
        if ub.is_top: return ua
        if ua.is_bottom or ub.is_bottom or ua.tag != ub.tag:
            return UNION_BOTTOM
        return self.lift(ua.tag, self.members[ua.tag].meet(ua.value, ub.value))

    def leq(self, a: Any, b: Any) -> bool:
        ua, ub = self._as_union(a), self._as_union(b)
        if ua.is_bottom or ub.is_top: return True
        if ub.is_bottom or ua.is_top or ua.tag != ub.tag: return False
        return self.members[ua.tag].leq(ua.value, ub.value)

    def const_u32(self, imm: int) -> UnionVal:
        return UnionVal(self.default_tag, self.members[self.default_tag].const_u32(imm))

    def const_u64(self, imm: int) -> UnionVal:
        return UnionVal(self.default_tag, self.members[self.default_tag].const_u64(imm))

    def const_bool(self, val: bool) -> UnionVal:
        return UnionVal(self.default_tag, self.members[self.default_tag].const_bool(val))

    def const_int_val(self, val: Any) -> Optional[int]:
        u = self._as_union(val)
        if u.is_bottom or u.is_top:
            return None
        return self.members[u.tag].const_int_val(u.value)

    def _unary(self, method: str, x: Any, *prefix_args: Any) -> UnionVal:
        ux = self._as_union(x)
        if ux.is_bottom: return UNION_BOTTOM
        if ux.is_top: return UNION_TOP
        res = getattr(self.members[ux.tag], method)(*prefix_args, ux.value)
        return self.lift(ux.tag, res)

    def _binary(self, method: str, a: Any, b: Any, *prefix_args: Any) -> UnionVal:
        ua, ub = self._as_union(a), self._as_union(b)
        if ua.is_bottom or ub.is_bottom: return UNION_BOTTOM
        if ua.is_top or ub.is_top: return UNION_TOP
        if ua.tag == ub.tag:
            res = getattr(self.members[ua.tag], method)(*prefix_args, ua.value, ub.value)
            return self.lift(ua.tag, res)
        op_key = prefix_args[0] if prefix_args else method
        handler = self.cross_ops.get((op_key, ua.tag, ub.tag)) or self.cross_ops.get((method, ua.tag, ub.tag))
        if handler is not None:
            res = handler(ua.value, ub.value)
            return res if isinstance(res, UnionVal) else UnionVal(self.default_tag, res)
        return UNION_TOP

    def cast_value(self, op: str, val: Any, val_type: str) -> UnionVal:
        ux = self._as_union(val)
        if ux.is_bottom: return UNION_BOTTOM
        if ux.is_top: return UNION_TOP
        return self.lift(ux.tag, self.members[ux.tag].cast_value(op, ux.value, val_type))

    def pack64(self, lo: Any, hi: Any) -> UnionVal: return self._binary("pack64", lo, hi)

    def split64(self, val64: Any) -> Tuple[UnionVal, UnionVal]:
        ux = self._as_union(val64)
        if ux.is_bottom: return (UNION_BOTTOM, UNION_BOTTOM)
        if ux.is_top: return (UNION_TOP, UNION_TOP)
        lo, hi = self.members[ux.tag].split64(ux.value)
        return (self.lift(ux.tag, lo), self.lift(ux.tag, hi))

    def int32_unary(self, op: str, x: Any) -> UnionVal: return self._unary("int32_unary", x, op)
    def int32_binary(self, op: str, a: Any, b: Any) -> UnionVal: return self._binary("int32_binary", a, b, op)
    def int64_unary(self, op: str, x: Any) -> UnionVal: return self._unary("int64_unary", x, op)
    def int64_binary(self, op: str, a: Any, b: Any) -> UnionVal: return self._binary("int64_binary", a, b, op)
    def icmp(self, op: str, a: Any, b: Any) -> UnionVal: return self._binary("icmp", a, b, op)
    def fp_unary(self, op: str, x: Any) -> UnionVal: return self._unary("fp_unary", x, op)
    def fp_binary(self, op: str, a: Any, b: Any) -> UnionVal: return self._binary("fp_binary", a, b, op)
    def fp_cmp(self, op: str, a: Any, b: Any) -> UnionVal: return self._binary("fp_cmp", a, b, op)
    def fp_is_nan(self, x: Any) -> UnionVal: return self._unary("fp_is_nan", x)
    def fp_is_inf(self, x: Any) -> UnionVal: return self._unary("fp_is_inf", x)
    def convert(self, op: str, x: Any) -> UnionVal: return self._unary("convert", x, op)
    def logical_not(self, a: Any) -> UnionVal: return self._unary("logical_not", a)
    def logical_and(self, a: Any, b: Any) -> UnionVal: return self._binary("logical_and", a, b)
    def logical_or(self, a: Any, b: Any) -> UnionVal: return self._binary("logical_or", a, b)

    def select(self, cond: Any, t_val: Any, f_val: Any) -> UnionVal:
        uc = self._as_union(cond)
        ut, uf = self._as_union(t_val), self._as_union(f_val)
        if not uc.is_bottom and not uc.is_top:
            if isinstance(uc.value, bool):
                return ut if uc.value else uf
            if uc.tag == ut.tag == uf.tag:
                return self.lift(ut.tag, self.members[ut.tag].select(uc.value, ut.value, uf.value))
        return self.join(ut, uf)


@dataclasses.dataclass(frozen=True)
class MultiSetVal:
    """
    Bounded finite multiset lattice element.

    `items` stores `(elem, multiplicity)` pairs for distinct elements;
    `is_top=True` represents the widened Top element; empty `items` with
    `is_top=False` represents Bottom.
    """

    items: Tuple[Tuple[Any, int], ...] = ()
    is_top: bool = False

    @property
    def is_bottom(self) -> bool:
        return (not self.is_top) and len(self.items) == 0

    @property
    def is_singleton(self) -> bool:
        return (not self.is_top) and len(self.items) == 1

    def elements(self) -> Tuple[Any, ...]:
        return tuple(e for e, _ in self.items)

    def multiplicity(self, elem: Any) -> int:
        for e, c in self.items:
            if e == elem:
                return c
        return 0

    def total_count(self) -> int:
        return sum(c for _, c in self.items)


MULTISET_BOTTOM = MultiSetVal((), is_top=False)
MULTISET_TOP = MultiSetVal((), is_top=True)


class MultiSetDomain(SsaDomain):
    """
    Bounded finite multiset (or powerset) domain over an element `SsaDomain`.

    Delegates unary operations element-wise and binary operations over the
    Cartesian product of operand elements, widening to `MULTISET_TOP` whenever
    the number of distinct elements exceeds `max_k`. When `track_counts=True`,
    multiplicities are preserved under join/sum and Cartesian products; when
    `track_counts=False`, all multiplicities are clamped to 1 (finite powerset).
    """

    def __init__(self, elem_domain: SsaDomain, max_k: int = 8, track_counts: bool = True):
        if max_k < 1:
            raise ValueError("MultiSetDomain max_k must be >= 1")
        self.elem_domain = elem_domain
        self.max_k = max_k
        self.track_counts = track_counts

    def _build(self, pairs: Sequence[Tuple[Any, int]]) -> MultiSetVal:
        merged: list[Tuple[Any, int]] = []
        for elem, cnt in pairs:
            if cnt <= 0:
                continue
            found = False
            for idx, (ex, ec) in enumerate(merged):
                if ex == elem:
                    merged[idx] = (ex, (ec + cnt) if self.track_counts else 1)
                    found = True
                    break
            if not found:
                if len(merged) >= self.max_k:
                    return MULTISET_TOP
                merged.append((elem, cnt if self.track_counts else 1))
        return MultiSetVal(tuple(merged), is_top=False)

    def singleton(self, elem: Any, count: int = 1) -> MultiSetVal:
        return self._build([(elem, count)])

    def from_elements(self, elems: Sequence[Any]) -> MultiSetVal:
        return self._build([(e, 1) for e in elems])

    def _as_ms(self, val: Any) -> MultiSetVal:
        return val if isinstance(val, MultiSetVal) else self.singleton(val)

    def bottom(self) -> MultiSetVal: return MULTISET_BOTTOM
    def top(self) -> MultiSetVal: return MULTISET_TOP
    def default_value(self) -> MultiSetVal: return MULTISET_TOP

    def join(self, a: Any, b: Any) -> MultiSetVal:
        ma, mb = self._as_ms(a), self._as_ms(b)
        if ma.is_bottom: return mb
        if mb.is_bottom: return ma
        if ma.is_top or mb.is_top: return MULTISET_TOP
        if self.track_counts:
            merged: list[Tuple[Any, int]] = list(ma.items)
            for eb, cb in mb.items:
                found = False
                for idx, (ea, ca) in enumerate(merged):
                    if ea == eb:
                        merged[idx] = (ea, max(ca, cb))
                        found = True
                        break
                if not found:
                    if len(merged) >= self.max_k:
                        return MULTISET_TOP
                    merged.append((eb, cb))
            return MultiSetVal(tuple(merged), is_top=False)
        return self._build(list(ma.items) + list(mb.items))

    def sum_union(self, a: Any, b: Any) -> MultiSetVal:
        """Multiset additive union (sums multiplicities of matching elements)."""
        ma, mb = self._as_ms(a), self._as_ms(b)
        if ma.is_top or mb.is_top: return MULTISET_TOP
        return self._build(list(ma.items) + list(mb.items))

    def meet(self, a: Any, b: Any) -> MultiSetVal:
        ma, mb = self._as_ms(a), self._as_ms(b)
        if ma.is_top: return mb
        if mb.is_top: return ma
        if ma.is_bottom or mb.is_bottom: return MULTISET_BOTTOM
        out: list[Tuple[Any, int]] = []
        for ea, ca in ma.items:
            cb = mb.multiplicity(ea)
            if cb > 0:
                out.append((ea, min(ca, cb) if self.track_counts else 1))
        return MultiSetVal(tuple(out), is_top=False)

    def leq(self, a: Any, b: Any) -> bool:
        ma, mb = self._as_ms(a), self._as_ms(b)
        if ma.is_bottom or mb.is_top: return True
        if ma.is_top or mb.is_bottom: return False
        for ea, ca in ma.items:
            if mb.multiplicity(ea) < ca:
                return False
        return True

    def const_u32(self, imm: int) -> MultiSetVal: return self.singleton(self.elem_domain.const_u32(imm))
    def const_u64(self, imm: int) -> MultiSetVal: return self.singleton(self.elem_domain.const_u64(imm))
    def const_bool(self, val: bool) -> MultiSetVal: return self.singleton(self.elem_domain.const_bool(val))

    def const_int_val(self, val: Any) -> Optional[int]:
        m = self._as_ms(val)
        if m.is_singleton:
            return self.elem_domain.const_int_val(m.items[0][0])
        return None

    def _lift_unary(self, fn: Callable[[Any], Any], x: Any) -> MultiSetVal:
        mx = self._as_ms(x)
        if mx.is_bottom: return MULTISET_BOTTOM
        if mx.is_top: return MULTISET_TOP
        return self._build([(fn(e), c) for e, c in mx.items])

    def _lift_binary(self, fn: Callable[[Any, Any], Any], a: Any, b: Any) -> MultiSetVal:
        ma, mb = self._as_ms(a), self._as_ms(b)
        if ma.is_bottom or mb.is_bottom: return MULTISET_BOTTOM
        if ma.is_top or mb.is_top: return MULTISET_TOP
        pairs = [(fn(ea, eb), ca * cb) for ea, ca in ma.items for eb, cb in mb.items]
        return self._build(pairs)

    def cast_value(self, op: str, val: Any, val_type: str) -> MultiSetVal:
        return self._lift_unary(lambda e: self.elem_domain.cast_value(op, e, val_type), val)

    def pack64(self, lo: Any, hi: Any) -> MultiSetVal:
        return self._lift_binary(self.elem_domain.pack64, lo, hi)

    def split64(self, val64: Any) -> Tuple[MultiSetVal, MultiSetVal]:
        m = self._as_ms(val64)
        if m.is_bottom: return (MULTISET_BOTTOM, MULTISET_BOTTOM)
        if m.is_top: return (MULTISET_TOP, MULTISET_TOP)
        los, his = [], []
        for e, c in m.items:
            lo, hi = self.elem_domain.split64(e)
            los.append((lo, c))
            his.append((hi, c))
        return (self._build(los), self._build(his))

    def int32_unary(self, op: str, x: Any) -> MultiSetVal:
        return self._lift_unary(lambda e: self.elem_domain.int32_unary(op, e), x)

    def int32_binary(self, op: str, a: Any, b: Any) -> MultiSetVal:
        return self._lift_binary(lambda ea, eb: self.elem_domain.int32_binary(op, ea, eb), a, b)

    def int64_unary(self, op: str, x: Any) -> MultiSetVal:
        return self._lift_unary(lambda e: self.elem_domain.int64_unary(op, e), x)

    def int64_binary(self, op: str, a: Any, b: Any) -> MultiSetVal:
        return self._lift_binary(lambda ea, eb: self.elem_domain.int64_binary(op, ea, eb), a, b)

    def icmp(self, op: str, a: Any, b: Any) -> MultiSetVal:
        return self._lift_binary(lambda ea, eb: self.elem_domain.icmp(op, ea, eb), a, b)

    def fp_unary(self, op: str, x: Any) -> MultiSetVal:
        return self._lift_unary(lambda e: self.elem_domain.fp_unary(op, e), x)

    def fp_binary(self, op: str, a: Any, b: Any) -> MultiSetVal:
        return self._lift_binary(lambda ea, eb: self.elem_domain.fp_binary(op, ea, eb), a, b)

    def fp_cmp(self, op: str, a: Any, b: Any) -> MultiSetVal:
        return self._lift_binary(lambda ea, eb: self.elem_domain.fp_cmp(op, ea, eb), a, b)

    def fp_is_nan(self, x: Any) -> MultiSetVal:
        return self._lift_unary(self.elem_domain.fp_is_nan, x)

    def fp_is_inf(self, x: Any) -> MultiSetVal:
        return self._lift_unary(self.elem_domain.fp_is_inf, x)

    def convert(self, op: str, x: Any) -> MultiSetVal:
        return self._lift_unary(lambda e: self.elem_domain.convert(op, e), x)

    def logical_not(self, a: Any) -> MultiSetVal:
        return self._lift_unary(self.elem_domain.logical_not, a)

    def logical_and(self, a: Any, b: Any) -> MultiSetVal:
        return self._lift_binary(self.elem_domain.logical_and, a, b)

    def logical_or(self, a: Any, b: Any) -> MultiSetVal:
        return self._lift_binary(self.elem_domain.logical_or, a, b)

    def select(self, cond: Any, t_val: Any, f_val: Any) -> MultiSetVal:
        mc, mt, mf = self._as_ms(cond), self._as_ms(t_val), self._as_ms(f_val)
        if mc.is_bottom:
            return MULTISET_BOTTOM
        if mc.is_singleton and isinstance(mc.items[0][0], bool):
            return mt if mc.items[0][0] else mf
        return self.join(mt, mf)
