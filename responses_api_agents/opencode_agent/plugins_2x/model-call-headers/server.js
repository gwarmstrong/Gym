// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// OpenCode 2.x plugin: per-call correlation headers. 2.2.5 does not expose the assistant message ID here.
export default {
  id: "nemo-gym-model-call-headers",
  setup: async (ctx) => {
    await ctx.session.hook("model.request", (event) => {
      const activation = process.env.NEMO_GYM_OPENCODE_ACTIVATION_ID
      if (activation) event.headers["X-Session-Id"] = `${activation}:${event.sessionID}`
      event.headers["X-OpenCode-Request-Kind"] = event.kind
    })
  },
}
