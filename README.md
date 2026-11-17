
## Build & Test Instructions

```bash
cmake -B build/                                   # Configure build directory
cmake --build build/                              # Build libvalhall2spirv.a, libmoc_hook.so, and all 4 test binaries
ctest --test-dir build --output-on-failure        # Run all unit and roundtrip tests
cmake --build build/ --target codegen             # Regenerate C++17 ISA/CSF headers and sources from XML specs
```
