// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include <cstdint>
#include <limits>

// Per-device admission of one more retained slot. The budget bounds the retained
// slots' cache bytes. On a CPU, where no device memory can be measured, the cached
// graph arenas are charged against the same budget, since it is the only guard. On
// an accelerator they are not: the free-memory check below already keeps room for
// one more slot and an arena as large as the largest observed shape, and charging
// them twice refused retention after any long prefill (a 1.9 GB prefill graph left
// a 2 GB budget no room for a 110 MB slot, so those conversations re-prefilled).
// This is admission, not a promise that an arbitrary future graph allocation
// cannot fail.
inline bool dsv41_retention_fits(uint64_t slot_bytes, uint64_t graph_bytes,
    uint64_t largest_graph, uint64_t retained_count, uint64_t budget,
    bool device_memory_known, uint64_t free_bytes, uint64_t reserve_bytes)
{
    const uint64_t charged_graphs = device_memory_known ? 0 : graph_bytes;
    if (!budget || charged_graphs > budget || retained_count == UINT64_MAX) return false;
    if (slot_bytes > (budget - charged_graphs) / (retained_count + 1)) return false;
    if (!device_memory_known) return true; // CPU: explicit byte budget only.
    if (slot_bytes > free_bytes) return false;
    free_bytes -= slot_bytes;
    if (largest_graph > free_bytes) return false;
    return reserve_bytes <= free_bytes - largest_graph;
}
