# Project state
Last updated: 2026-10-15
## Works
- The Next.js site (App Router, TypeScript, plain CSS) is live on Vercel at https://ai-workshop-blush-nine.vercel.app
- Slice 1 is live (PR #4, merged 2026-10-09): sign-up at /signup, log-in at /login, and a protected /picks page that shows "Signed in as <email>" and a Log out button. Logged-out visits to /picks go to /login. A wrong password shows "Wrong email or password." Sessions survive closing the browser. Checked on the live site and the preview.
- Slice 2 is done (PR #6, merged; checked on the PR #9 preview and then on the live site). A signed-in visitor can open /picks and see every row of the predictions table, soonest date first (same date: away team A to Z), each with labeled date, away team, home team, predicted winner and win percent, F5 winner, and projected total runs. It reads the table on every visit, so new rows show up on refresh without a redeploy. Logged-out visitors who go to /picks are sent to /login.
- Slice 3 is done (PR #12, merged 2026-10-15; checked on the PR #12 preview): Past, Upcoming, and Future tabs on /picks. Tabs split games by today's date in Hawaii time. With no tab picked, Upcoming opens, or Future if no game is dated today. Clicking a tab adds ?tab=past, ?tab=upcoming, or ?tab=future to the address. No database change.
- The predictions table exists in Supabase with row-level security on. Its only rule lets signed-in users read; nobody can add, change, or delete rows through the website. Columns: id, game_date, away_team, home_team, predicted_winner, win_pct (0 to 100), f5_winner, projected_total, created_at.
- The site uses Supabase Auth. Test accounts tester1@example.com, tester2@example.com, and tester3@gmail.com exist. Future test accounts use @example.com addresses only.
- The GitHub repo Niko80896/AI-Workshop exists, default branch main.
- The homepage is still the personal site from Session 1, unchanged.
## Broken or flaky
- Supabase project is new (created 2026-10-08 because the Session 1 one was missing); linking from Session 1 not confirmed with instructor.
- If Supabase refuses a login for a reason other than a wrong password (for example too many attempts), /login shows Supabase's own message instead of "Wrong email or password."
- Real predictions and test rows share one table. Slice 2's criteria check the order of the test rows, not the total count.
- The predictions table holds real games loaded by hand from the model: White Sox at Guardians (2026-10-10) and Brewers at Dodgers (2026-10-14). The Test Past and Test Future rows from checking Slice 3 are still there; Niko is deleting them after checking the live site. The Slice 2 test rows are gone.
- PR #7 (Docs: slice 2 done, slice 3 active) must never be merged because its formatting was broken (it dropped the # heading markers). As of 2026-10-15 it is still open as a draft on GitHub and should be closed unmerged.
- Nothing else known to be broken.
## Environment notes
- Stack: Next.js 16 (App Router), TypeScript, plain CSS, Supabase, Vercel.
- Repo: Niko80896/AI-Workshop. Default branch: main.
- Live site: https://ai-workshop-blush-nine.vercel.app (Vercel project: ai-workshop).
- Environment variables NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY are set in Vercel for Production and Preview.
- Email confirmation is off in Supabase. Site URL is the live address; the one Redirect URL is https://ai-workshop-blush-nine.vercel.app/**
- The preview and the live site share one Supabase project, so test accounts created on a preview also exist on the live site.
- The prediction model runs outside this repo. Niko loads its output file (one row per game) into a Supabase table by hand. The website only reads and displays it and never runs the model.
- Branch claude/tender-fermat-a5sg8b holds old model code from a mistaken session. Never merge it.
- Every branch pushed to GitHub gets a Vercel preview link on its pull request. Merging to main deploys the live site.
- Keys and passwords live only in Vercel > ai-workshop > Settings > Environment Variables. Never in chat, prompts, or repo files.
- Test accounts are tester1@example.com, tester2@example.com, and tester3@gmail.com (Niko's own test account). Any new test accounts use @example.com addresses only.
## Next session
- Slice 3 (Past, Upcoming, Future tabs) is DONE.
- Slice 4 (Like and dislike) is PENDING. Its Setup needs the Slice 2 test rows (Test Cubs, Test Hawks, Test Lions, Test Bears) added back to the predictions table first.
