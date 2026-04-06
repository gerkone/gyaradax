#pragma once
#include <cstddef>

// Forward declarations for v5 LTO callback fatbins.
// Definitions live in bracket_v5_lto_fatbins.cu.
extern unsigned long long bracket_v5_z2z_load_cb_fatbin[];
extern unsigned long long bracket_v5_d2z_load_cb_fatbin[];
extern unsigned long long bracket_v5_store_cb_fatbin[];

extern const size_t bracket_v5_z2z_load_cb_fatbin_bytes;
extern const size_t bracket_v5_d2z_load_cb_fatbin_bytes;
extern const size_t bracket_v5_store_cb_fatbin_bytes;
