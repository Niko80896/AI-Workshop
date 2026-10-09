# Project state
Last updated: 2026-10-09
## Works
- The Next.js site (App Router, TypeScript, plain CSS) is live on Vercel at https://ai-workshop-blush-nine.vercel.app
- Slice 1 is live (PR #4, merged 2026-10-09): sign-up at /signup, log-in at /login, and a protected /picks page that shows "Signed in as <email>" and a Log out button. Logged-out visits to /picks go to /login. A wrong password shows "Wrong email or password." Sessions survive closing the browser. Checked on the live site and the preview.
- The site uses Supabase Auth. Test accounts tester1@example.com and tester2@example.com exist.
- The GitHub repo Niko80896/AI-Workshop exists, default branch main.
- The homepage is still the personal site from Session 1, unchanged.
## Broken or flaky
- Supabase project is new (created 2026-10-08 because the Session 1 one was missing); linking from Session 1 not confirmed with instructor.
- If Supabase refuses a login for a reason other than a wrong password (for example too many attempts), /login shows Supabase's own message instead of "Wrong email or password."
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
- Test accounts use fake emails only (tester1@example.com and similar).
## Next session
- Slice 2 (Predictions page) is ACTIVE. See roadmap.md.
- Claude Code will write the SQL for the predictions table, with row-level security so only signed-in users can read it. Niko runs it in the Supabase SQL editor and loads the test rows listed in roadmap.md. Win percent is stored as a number from 0 to 100 (for example 62.5). Delete the Test rows before loading real predictions.
