# X/Twitter Account Monitor Skill

> Automated tracking and forwarding of X/Twitter account updates via RSS polling.

## Overview

This skill provides a lightweight, proxy-friendly approach to monitor X accounts without API access, using nitter.net RSS feeds and periodic polling via cron jobs.

## Use Cases

- Track AI researchers' latest thoughts (@karpathy, @ylecun, etc.)
- Monitor product launches and announcements
- Follow conference/live event coverage
- Stay updated on fast-moving discussions

## Prerequisites

- Network proxy (if required for access)
- `web_fetch` or `exec` with `curl`
- `cron` - For scheduled execution
- `message` - For notifications

## Data Source

### nitter.net RSS

**Base URL:** `https://nitter.net/{username}/rss`

**Example:**
- https://nitter.net/karpathy/rss
- https://nitter.net/stingning/rss
- https://nitter.net/arankomatsuzaki/rss

**Why nitter.net:**
- No authentication required
- RSS format (structured, parseable)
- Works through proxies
- Respects rate limits
- Alternative to X API (which requires developer account)

**⚠️ Limitations:**
- nitter.net instance may be unstable
- ~5-15 minute delay from original post
- No direct media attachments (links only)
- Instance may block aggressive polling

## Workflow

### Phase 1: Account Configuration

#### 1.1 Define Account List

Create a configuration file or inline list:

```json
{
  "accounts": [
    {
      "username": "karpathy",
      "name": "Andrej Karpathy",
      "category": "AI Research",
      "priority": "high"
    },
    {
      "username": "stingning",
      "name": "Ning Ding",
      "category": "AI Research",
      "priority": "high"
    },
    {
      "username": "arankomatsuzaki",
      "name": "Aran Komatsuzaki",
      "category": "AI Research",
      "priority": "medium"
    }
  ]
}
```

#### 1.2 Set Proxy (if needed)

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
```

### Phase 2: Fetch & Parse

#### 2.1 Fetch RSS Feed

```bash
# Using curl with proxy
RSS_CONTENT=$(curl -sL "https://nitter.net/{username}/rss" \
  -H "User-Agent: Mozilla/5.0" \
  --proxy http://127.0.0.1:7890 2>/dev/null)
```

Or via `web_fetch`:
```bash
web_fetch "https://nitter.net/{username}/rss" --extractMode text
```

#### 2.2 Parse RSS Structure

**RSS Format:**
```xml
<item>
  <title>Tweet content (may be truncated)</title>
  <description><![CDATA[Full HTML content]]></description>
  <pubDate>Mon, 16 Feb 2026 12:00:00 GMT</pubDate>
  <guid isPermaLink="false">tweet_id_string</guid>
  <link>https://nitter.net/username/status/tweet_id</link>
</item>
```

**Extract fields:**
- `title` - Tweet text (truncated if long)
- `description` - Full content with HTML
- `pubDate` - Publication date
- `guid` - Unique tweet ID
- `link` - Link to tweet

### Phase 3: State Management

#### 3.1 Track Last Seen Tweet

Store state to detect new tweets:

```bash
STATE_DIR="$HOME/.twitter-monitor"
mkdir -p "$STATE_DIR"

# Save last seen tweet ID
echo "$TWEET_ID" > "$STATE_DIR/{username}-last-id"
```

#### 3.2 Detect New Tweets

```bash
LAST_ID=$(cat "$STATE_DIR/{username}-last-id" 2>/dev/null || echo "")
CURRENT_ID=$(echo "$RSS" | grep -oP '(?<=<guid isPermaLink="false">)[^<]+' | head -1)

if [ "$CURRENT_ID" != "$LAST_ID" ] && [ -n "$LAST_ID" ]; then
  # New tweet detected!
  echo "New tweet from @{username}"
fi
```

### Phase 4: Format & Deliver

#### 4.1 Format Notification

**Basic Format:**
```markdown
🐦 **New tweet from @{display_name}** (@{username})

{text}

🔗 {link}
🕒 {pub_date}
```

**With Commentary (锐评):**
```markdown
🐦 **New tweet from @{display_name}** (@{username})

{text}

🔗 {link}
🕒 {pub_date}

---
**💡 Quick Take:** {1-2 sentence commentary on significance, technical insight, or context}
```

#### 4.2 When to Add Commentary

| Scenario | Action |
|----------|--------|
| Technical announcement | Explain significance briefly |
| Paper link shared | Note if it's related to user's interests |
| Industry news | Add context on impact |
| Casual/personal tweet | Skip commentary |
| Multiple tweets | Batch with summary only |

**Guidelines:**
- Keep it brief (1-2 sentences)
- Connect to user's research interests if relevant
- Avoid commentary on purely personal tweets
- Don't force commentary if there's nothing meaningful to add

#### 4.3 Send Notification

```bash
# Via message tool
message send \
  --channel feishu \
  --to "{user_id}" \
  --text "$NOTIFICATION"
```

**Important:** Only send if new tweets detected. Skip silently if no updates.

### Phase 5: Error Handling

#### 5.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Empty RSS | nitter instance down | Retry with fallback instance |
| Parse error | XML malformed | Use regex fallback |
| Rate limited | Too frequent polling | Reduce frequency (30min+) |
| Proxy error | Network issue | Check proxy connection |

#### 5.2 Fallback Instances

If primary nitter.net fails, try alternatives:
- nitter.net (primary)
- nitter.cz
- nitter.privacydev.net
- nitter.projectsegfault.com

**Check instance status:** https://github.com/zedeus/nitter/wiki/Instances

## Cron Job Setup

### Create Monitor Job

```bash
openclaw cron add \
  --name "Twitter Monitor - {collection_name}" \
  --cron "*/30 * * * *" \
  --tz "Asia/Shanghai" \
  --session isolated \
  --message "Check RSS feeds for accounts: {list}. For each account: 1) Fetch nitter.net/{username}/rss, 2) Compare with stored state, 3) If new tweet ONLY: format with original X link, add brief commentary if it's technical/interesting (1-2 sentences), and send to {target_user}. 4) If no new tweets: skip silently (don't send anything). Handle errors gracefully." \
  --announce \
  --channel feishu \
  --to "{user_id}"
```

**Key behaviors:**
- ✅ Only forward NEW tweets (state comparison)
- ✅ Include original X link (not nitter.net)
- ✅ Optional brief commentary for technical/interesting tweets
- ✅ Skip silently if no new tweets (no "no updates" message)

### Polling Frequency Guidelines

| Frequency | Use Case | Risk |
|-----------|----------|------|
| 15 min | Live events | High (rate limit) |
| 30 min | Active tracking | Medium |
| 1 hour | Daily updates | Low |
| 6 hours | Weekly digest | Very low |

**Recommended:** 30 minutes for active accounts

## Output Examples

### Single Tweet Notification (With Commentary)

```markdown
🐦 **New tweet from Andrej Karpathy** (@karpathy)

Congrats on the launch @simile_ai! Simile is working on a really interesting, imo under-explored dimension of LLMs...

🔗 https://x.com/karpathy/status/188972345188580788
🕒 Thu, 12 Feb 2026 20:12:52 GMT

---
**💡 Quick Take:** Karpathy is betting on "population simulation" as the next paradigm beyond single-personality chatbots. Interesting that he's backing this with investment—worth watching how Simile develops the entropy management problem.
```

### Single Tweet Notification (No Commentary)

```markdown
🐦 **New tweet from Ning Ding** (@stingning)

When your project gets too good, your next PR is either a legal letter or a payroll.

🔗 https://x.com/stingning/status/...
🕒 Thu, 12 Feb 2026 18:30:00 GMT
```

*(No commentary added—casual observation tweet)*

### Batch Summary (Multiple New Tweets)

```markdown
🐦 **Twitter Updates - Last 30min**

**@karpathy** (2 new)
1. Simile AI investment announcement
2. Micrograd code simplification

**@stingning** (1 new)
1. When your project gets too good...

**@arankomatsuzaki** (0 new)

_Total: 3 new tweets_

---
**💡 Quick Takes:**
- **Karpathy:** Two tweets today—one on AI investment (Simile's population simulation) and one on code minimalism (200-line micrograd). Both show his focus on elegant abstractions.
```

**Note:** No notification sent if 0 new tweets across all accounts.

## Complete Script Template

```bash
#!/bin/bash
# twitter-monitor.sh - With commentary support

ACCOUNTS=("karpathy" "stingning" "arankomatsuzaki")
TARGET_USER="ou_168ea1a1162ad66582b40ec15e5a2950"
STATE_DIR="$HOME/.twitter-monitor"
HAS_NEW_TWEETS=false

# Set proxy
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

mkdir -p "$STATE_DIR"

for username in "${ACCOUNTS[@]}"; do
    RSS_URL="https://nitter.net/${username}/rss"
    STATE_FILE="${STATE_DIR}/${username}-last-id"
    
    # Fetch RSS
    RSS=$(curl -sL "$RSS_URL" -H "User-Agent: Mozilla/5.0" 2>/dev/null)
    
    if [ -z "$RSS" ]; then
        echo "Failed to fetch $username"
        continue
    fi
    
    # Extract current tweet ID
    CURRENT_ID=$(echo "$RSS" | grep -oP '(?<=<guid isPermaLink="false">)[^<]+' | head -1)
    
    # Get last known ID
    LAST_ID=$(cat "$STATE_FILE" 2>/dev/null || echo "")
    
    # Check if new
    if [ "$CURRENT_ID" != "$LAST_ID" ] && [ -n "$LAST_ID" ]; then
        HAS_NEW_TWEETS=true
        
        # Extract tweet details
        TITLE=$(echo "$RSS" | grep -oP '(?<=<title>)[^<]+' | head -1 | sed 's/&quot;/"/g; s/&amp;/\&/g')
        LINK=$(echo "$RSS" | grep -oP '(?<=<link>)[^<]+' | head -1 | sed 's|nitter.net|x.com|')
        DATE=$(echo "$RSS" | grep -oP '(?<=<pubDate>)[^<]+' | head -1)
        
        # Format message (basic)
        MESSAGE="🐦 **New tweet from @${username}**

${TITLE}

🔗 ${LINK}
🕒 ${DATE}"
        
        # Add commentary for technical/interesting tweets
        # (In real implementation, AI would analyze content here)
        if echo "$TITLE" | grep -qiE "(paper|arxiv|llm|model|training|research|launch|product)"; then
            MESSAGE="${MESSAGE}

---
**💡 Quick Take:** [AI generates 1-2 sentence commentary on significance]"
        fi
        
        # Send notification
        echo "$MESSAGE"
        # openclaw message send --channel feishu --to "$TARGET_USER" --text "$MESSAGE"
    fi
    
    # Update state
    echo "$CURRENT_ID" > "$STATE_FILE"
done

# Only send summary if there were new tweets
if [ "$HAS_NEW_TWEETS" = false ]; then
    echo "No new tweets - skipping notification"
    exit 0
fi
```

**Key behaviors implemented:**
- Only processes NEW tweets (state comparison)
- Uses original X links (replaces nitter.net with x.com)
- Adds commentary only for technical/AI-related content
- Skips silently if no new tweets

## Troubleshooting

### "Failed to fetch"

**Check:**
1. Proxy is set: `echo $http_proxy`
2. nitter.net is accessible: `curl -I https://nitter.net`
3. Try alternative instance

### "Empty RSS content"

**Causes:**
- Account doesn't exist or is private
- nitter instance is overloaded
- Account has no recent tweets

### Duplicate notifications

**Fix:** Ensure state file is being written correctly after each check.

### Missing tweets

**Causes:**
- Polling interval too long (tweet missed between checks)
- nitter instance lag
- Account posted multiple tweets in quick succession

## Related Skills

- `daily-ai-papers` - Combine with Twitter for comprehensive AI monitoring
- `web_fetch` - For additional web scraping
- `cron` - For scheduling

## References

- nitter.net: https://nitter.net
- Nitter instances list: https://github.com/zedeus/nitter/wiki/Instances
- RSS specification: https://www.rssboard.org/rss-specification

---

*Skill created: 2026-02-16*
