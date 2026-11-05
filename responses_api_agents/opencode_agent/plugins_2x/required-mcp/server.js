// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFile, rename, writeFile } from "node:fs/promises"

// OpenCode 2.x plugin, the counterpart of required_mcp.mjs. OpenCode leaves an MCP server that fails to connect out
// of the tool list and keeps running. Before the first model call, record each grant's status and stop the run when a
// required grant is missing: a session hook that throws fails the execution.
export default {
  id: "nemo-gym-required-mcp",
  setup: async (ctx) => {
    const grants = JSON.parse(await readFile(new URL("../mcp-grants.json", import.meta.url), "utf8"))

    const check = async () => {
      // 2.2.5 answers {location, data: [{name, status: {status, error?}}]}.
      const listed = await ctx.mcp.list().catch(() => undefined)
      const servers = new Map((listed?.data ?? []).map((entry) => [entry.name, entry.status]))
      const statuses = {}
      for (const name of [...grants.required, ...grants.optional]) {
        const status = servers.get(name)
        statuses[name] = {
          status: status?.status ?? "unavailable",
          ...(status?.error ? { error: String(status.error).slice(0, 2000) } : {}),
        }
      }
      // Replace the staged file rather than write into it: the adapter's upload may own it as another user.
      const staged = new URL(`../mcp-status.json.${process.pid}`, import.meta.url)
      await writeFile(staged, JSON.stringify(statuses))
      await rename(staged, new URL("../mcp-status.json", import.meta.url))
      return grants.required.filter((name) => statuses[name].status !== "connected")
    }

    let missing
    await ctx.session.hook("model.request", async () => {
      if (missing === undefined) missing = await check()
      // Fail every later model call too, in case OpenCode continues after the first failure.
      if (missing.length) throw new Error("Required Gym MCP tools could not be initialized: " + missing.join(", "))
    })
  },
}
