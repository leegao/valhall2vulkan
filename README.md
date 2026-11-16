
## Build & Test Instructions

```bash
make codegen   # Regenerate C++17 ISA/CSF headers, sources, and formal semantics docs from XML specs
make all       # Build build/libvalhall2spirv.a, build/libmoc_hook.so, and all 4 test binaries
make test      # Run test_isa_decoder, test_csf_descriptors, test_moc_roundtrip_compute, and test_moc_roundtrip_graphics
```
