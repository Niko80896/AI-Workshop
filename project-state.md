# Project state
Last updated: 2026-10-08
## Works
- The Next.js site (App Router, TypeScript, plain CSS) is live on Vercel at https://ai-workshop-blush-nine.vercel.app
- The GitHub repo Niko80896/AI-Workshop exists, default branch main.
- A Supabase project exists. It was created on 2026-10-08 because the one from Session 1 was missing. The site does not use it yet.
## Broken or flaky
- Supabase project is new; linking from Session 1 not confirmed with instructor.
- Nothing else known to be broken.
## Environment notes
- Stack: Next.js (App Router), TypeScript, plain CSS, Supabase, Vercel.
- Repo: Niko80896/AI-Workshop. Default branch: main.
- Live site: https://ai-workshop-blush-nine.vercel.app (Vercel project: ai-workshop).
- Environment variables NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY were added in Vercel on 2026-10-08 for Production and Preview. New values apply only to deployments made after they were added.
- Email confirmation is off in Supabase. Niko turned it off on 2026-10-08; check that Authentication > Sign In / Providers > Email shows Confirm email as off before testing Slice 1.
- The prediction model runs outside this repo. Niko loads its output file (one row per game) into a Supabase table by hand. The website only reads and displays it and never runs the model.
- Branch claude/tender-fermat-a5sg8b holds old model code from a mistaken session. Never merge it.
- Every branch pushed to GitHub gets a Vercel preview link on its pull request. Merging to main deploys the live site.
- Keys and passwords live only in Vercel > ai-workshop > Settings > Environment Variables. Never in chat, prompts, or repo files.
- Test accounts use fake emails only (tester1@example.com and similar).
## Next session
- Slice 1 (sign up and log in) is ACTIVE. See roadmap.md.
- Confirm in Supabase that email confirmation is off.
- Expect Claude Code to ask permission to add the Supabase client library. That is a new dependency and needs an explicit yes.
- Before Slice 2: Claude Code will write the SQL for the predictions table; Niko runs it in the Supabase SQL editor and loads the test rows listed in roadmap.md. Win percent is stored as a number from 0 to 100 (for example 62.5). Delete the Test rows before loading real predictions.
