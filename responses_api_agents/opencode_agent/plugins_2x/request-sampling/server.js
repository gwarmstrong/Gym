// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// OpenCode 2.x plugin: apply the Gym request's sampling to every model request of every agent.
import { readFile } from "node:fs/promises"

export default {
  id: "nemo-gym-request-sampling",
  setup: async (ctx) => {
    const sampling = JSON.parse(await readFile(new URL("../sampling.json", import.meta.url), "utf8"))
    await ctx.session.hook("context", (event) => {
      Object.assign(event.options, sampling)
    })
  },
}
