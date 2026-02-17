# Cron Jobs Interface Skill

> Complete guide to OpenClaw's cron scheduling system for automated task execution.

## Overview

OpenClaw's cron system allows scheduling recurring or one-time tasks that:
- Run at specified times or intervals
- Execute in main or isolated sessions
- Deliver output to channels (announce) or webhooks
- Persist across restarts

## Concepts

### Job Components

| Component | Description | Required |
|-----------|-------------|----------|
| **name** | Human-readable identifier | ✅ |
| **schedule** | When to run (at/every/cron) | ✅ |
| **sessionTarget** | main or isolated | ✅ |
| **payload** | What to execute | ✅ |
| **delivery** | Where to send output | Optional |
| **wakeMode** | When to wake (now/next-heartbeat) | Optional |

### Session Types

#### Main Session (`sessionTarget: "main"`)
- Runs during heartbeat with main context
- Shares conversation history
- Payload: `systemEvent`
- Use for: Context-aware tasks that need prior conversation

#### Isolated Session (`sessionTarget: "isolated"`)
- Runs in dedicated `cron:<jobId>` session
- Fresh context each time
- Payload: `agentTurn`
- Use for: Background tasks, noisy/frequent jobs

### Schedule Types

| Type | Format | Use Case |
|------|--------|----------|
| **at** | ISO 8601 timestamp | One-time reminder |
| **every** | Milliseconds interval | Fixed periodic tasks |
| **cron** | Cron expression | Complex schedules |

## CLI Usage

### Add a Job

#### One-shot Reminder (Main Session)
```bash
openclaw cron add \
  --name "Send reminder" \
  --at "2026-02-01T16:00:00Z" \
  --session main \
  --system-event "Check the cron docs" \
  --wake now \
  --delete-after-run
```

#### Recurring Job (Isolated Session)
```bash
openclaw cron add \
  --name "Morning brief" \
  --cron "0 7 * * *" \
  --tz "America/Los_Angeles" \
  --session isolated \
  --message "Summarize overnight updates." \
  --announce \
  --channel feishu \
  --to "user_id"
```

#### Every N Minutes
```bash
openclaw cron add \
  --name "Health check" \
  --every "30m" \
  --session main \
  --system-event "Run system health check"
```

### List Jobs
```bash
openclaw cron list
```

Output shows:
- Job ID, name, schedule
- Next run time
- Last run status
- Consecutive errors

### Run Job Manually
```bash
# Force run (default)
openclaw cron run <job-id>

# Only run if due
openclaw cron run <job-id> --due
```

### Edit Job
```bash
openclaw cron edit <job-id> \
  --message "Updated prompt text" \
  --cron "0 9 * * *"
```

### Remove Job
```bash
openclaw cron remove <job-id>
```

### View Run History
```bash
openclaw cron runs --id <job-id> --limit 20
```

## Tool Call API

### cron.add

#### One-shot, Main Session
```json
{
  "name": "Reminder",
  "schedule": { "kind": "at", "at": "2026-02-01T16:00:00Z" },
  "sessionTarget": "main",
  "wakeMode": "now",
  "payload": { "kind": "systemEvent", "text": "Reminder text" },
  "deleteAfterRun": true
}
```

#### Recurring, Isolated, with Delivery
```json
{
  "name": "Morning brief",
  "schedule": { 
    "kind": "cron", 
    "expr": "0 7 * * *", 
    "tz": "America/Los_Angeles" 
  },
  "sessionTarget": "isolated",
  "wakeMode": "next-heartbeat",
  "payload": {
    "kind": "agentTurn",
    "message": "Summarize overnight updates."
  },
  "delivery": {
    "mode": "announce",
    "channel": "feishu",
    "to": "user_id",
    "bestEffort": true
  }
}
```

#### Every N Minutes
```json
{
  "name": "Health check",
  "schedule": { "kind": "every", "everyMs": 1800000 },
  "sessionTarget": "main",
  "payload": { 
    "kind": "systemEvent", 
    "text": "Run health check" 
  }
}
```

### cron.update
```json
{
  "jobId": "job-123",
  "patch": {
    "enabled": false,
    "schedule": { "kind": "every", "everyMs": 3600000 }
  }
}
```

### cron.remove
```json
{ "jobId": "job-123" }
```

### cron.run
```json
{ 
  "jobId": "job-123",
  "mode": "force"  // or "due"
}
```

### cron.list
```json
{}
```

### cron.runs
```json
{
  "jobId": "job-123",
  "limit": 50
}
```

## Delivery Modes

### announce
- Sends output to specified channel
- Posts brief summary to main session
- Default for isolated jobs

```json
{
  "delivery": {
    "mode": "announce",
    "channel": "feishu",
    "to": "user_id"
  }
}
```

### webhook
- POSTs payload to URL
- No channel delivery
- No main session summary

```json
{
  "delivery": {
    "mode": "webhook",
    "to": "https://example.com/webhook"
  }
}
```

### none
- No external delivery
- Internal only

```json
{
  "delivery": { "mode": "none" }
}
```

## Channel Targets

### Feishu
- User: `ou_xxxxxx`
- Chat: `chat_xxxxxx`
- Omit target to use current conversation

### Slack
- Channel: `channel:C1234567890`
- User: `user:U1234567890`

### Discord
- Channel: `channel:1234567890`
- User: `user:1234567890`

### Telegram
- Chat: `1234567890`
- Topic: `-1001234567890:topic:123`

### WhatsApp
- Phone: `+15551234567`

## Cron Expression Reference

### Format: `minute hour day month weekday`

| Expression | Meaning |
|------------|---------|
| `0 8 * * *` | Every day at 8:00 AM |
| `*/30 * * * *` | Every 30 minutes |
| `0 */6 * * *` | Every 6 hours |
| `0 9 * * 1` | Every Monday at 9:00 AM |
| `0 0 1 * *` | First day of month |

### Timezones

Always specify timezone for clarity:
```json
{ "tz": "Asia/Shanghai" }
{ "tz": "America/New_York" }
{ "tz": "UTC" }
```

## Common Patterns

### Daily Task (Beijing Time)
```bash
openclaw cron add \
  --name "Daily AI Papers" \
  --cron "0 8 * * *" \
  --tz "Asia/Shanghai" \
  --session isolated \
  --message "Fetch and curate AI papers" \
  --announce \
  --channel feishu \
  --to "user_id"
```

### Frequent Polling
```bash
openclaw cron add \
  --name "Twitter Monitor" \
  --cron "*/30 * * * *" \
  --tz "Asia/Shanghai" \
  --session isolated \
  --message "Check Twitter RSS feeds" \
  --announce \
  --channel feishu \
  --to "user_id"
```

### One-time Reminder
```bash
openclaw cron add \
  --name "Meeting reminder" \
  --at "20m" \
  --session main \
  --system-event "Meeting starts in 10 minutes" \
  --wake now
```

### Model Override
```bash
openclaw cron add \
  --name "Deep analysis" \
  --cron "0 2 * * 0" \
  --session isolated \
  --message "Weekly deep analysis" \
  --model "opus" \
  --thinking high \
  --announce
```

## Error Handling

### Consecutive Errors
- After failures, exponential backoff applies: 30s, 1m, 5m, 15m, 60m
- Resets after successful run

### Job States
| State | Meaning |
|-------|---------|
| ok | Last run succeeded |
| error | Last run failed |
| skipped | Run was skipped |
| running | Currently executing |

### Debugging Failed Jobs

1. **Check run history:**
   ```bash
   openclaw cron runs --id <job-id> --limit 10
   ```

2. **Verify schedule:**
   ```bash
   openclaw cron list
   ```

3. **Run manually:**
   ```bash
   openclaw cron run <job-id>
   ```

4. **Check logs:**
   - Run history: `~/.openclaw/cron/runs/<jobId>.jsonl`

## Storage & Persistence

- Jobs: `~/.openclaw/cron/jobs.json`
- Run history: `~/.openclaw/cron/runs/<jobId>.jsonl`
- Survives Gateway restarts
- Manual edits only safe when Gateway stopped

## Configuration

```json5
// openclaw.config.json
{
  cron: {
    enabled: true,
    store: "~/.openclaw/cron/jobs.json",
    maxConcurrentRuns: 1,
    webhook: "https://example.com/webhook",
    webhookToken: "bearer-token"
  }
}
```

Disable via environment:
```bash
OPENCLAW_SKIP_CRON=1
```

## Best Practices

### DO:
- ✅ Use descriptive job names
- ✅ Always specify timezone
- ✅ Use isolated sessions for frequent/noisy tasks
- ✅ Set `bestEffort: true` for delivery to avoid job failures
- ✅ Use `deleteAfterRun` for one-shot jobs
- ✅ Monitor consecutive errors

### DON'T:
- ❌ Create too many frequent jobs (rate limits)
- ❌ Use main session for long-running tasks
- ❌ Forget to set timezone (defaults to host timezone)
- ❌ Edit jobs.json while Gateway is running

## Troubleshooting

### "Nothing runs"
- Check `cron.enabled` in config
- Verify Gateway is running continuously
- Check for `OPENCLAW_SKIP_CRON=1`

### "Job not executing at expected time"
- Verify timezone matches expectation
- Check system time on Gateway host
- Review cron expression

### "Delivery fails"
- Check channel configuration
- Verify target ID is correct
- Use `bestEffort: true` to prevent job failure

## Related Skills

- `daily-ai-papers` - Uses cron for daily execution
- `x-twitter-monitor` - Uses cron for periodic polling
- `web_fetch` - Often used in cron jobs
- `message` - Delivery mechanism

## References

- Full docs: https://docs.openclaw.ai/automation/cron-jobs
- Cron syntax: https://crontab.guru/
- Nitter instances: https://github.com/zedeus/nitter/wiki/Instances

---

*Skill created: 2026-02-16*
