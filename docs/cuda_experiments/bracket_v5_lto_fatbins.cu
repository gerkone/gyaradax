// Single translation unit that owns the v5 LTO callback fatbin definitions.
// Both cufft_graph_bracket.cu and cufft_graph_bracket_fp64.cu reference these
// via extern declarations in bracket_v5_lto_fatbins_decl.h.

#include "bracket_v5_z2z_load_cb_fatbin.h"
#include "bracket_v5_d2z_load_cb_fatbin.h"
#include "bracket_v5_store_cb_fatbin.h"

extern const size_t bracket_v5_z2z_load_cb_fatbin_bytes = sizeof(bracket_v5_z2z_load_cb_fatbin);
extern const size_t bracket_v5_d2z_load_cb_fatbin_bytes = sizeof(bracket_v5_d2z_load_cb_fatbin);
extern const size_t bracket_v5_store_cb_fatbin_bytes     = sizeof(bracket_v5_store_cb_fatbin);
