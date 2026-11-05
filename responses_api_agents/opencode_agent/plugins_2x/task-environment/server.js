// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// OpenCode 2.x plugin: give task shells the task's HOME/XDG directories and write the adapter's receipt.
import { readFile, writeFile } from "node:fs/promises"

export default {
  id: "nemo-gym-task-environment",
  setup: async (ctx) => {
    const env = JSON.parse(await readFile(new URL("../task-env.json", import.meta.url), "utf8"))
    for (const key of ["HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"]) {
      if (typeof env?.[key] !== "string" || !env[key].startsWith("/")) throw new Error(`Invalid task directory: ${key}`)
    }
    await ctx.shell.hook("create.before", (event) => {
      Object.assign(event.env, env)
    })
    await writeFile(new URL("../task-environment.ready", import.meta.url), "ready\n")
  },
}
