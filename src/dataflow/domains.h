#pragma once

/// Umbrella header for the composable SFINAE abstract-interpretation domain library.
/// Individual domain components live under `src/dataflow/domains/` and may be included
/// directly when only a subset of combinators is needed.

#include "domains/traits.h"
#include "domains/flat.h"
#include "domains/mixins.h"
#include "domains/containers.h"
#include "domains/bitset.h"
#include "domains/finite_set.h"
#include "domains/union.h"
#include "domains/lane.h"
