October
Stack

* Next.js 16 (App Router), TypeScript, plain CSS (no CSS framework).
* Supabase for sign-in and the database.
* Deployed on Vercel (project: ai-workshop). Live site: https://ai-workshop-blush-nine.vercel.app
* GitHub repo: Niko80896/AI-Workshop. Default branch: main.
* Every pushed branch gets a Vercel preview link on its pull request. Merging to main deploys the live site.
* The site reads two environment variables, set in Vercel for Production and Preview: NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY. Use exactly these names.
* The MLB prediction model runs outside this repo. The site only reads and displays its output from a Supabase table.
* Database table public.predictions: id (bigint identity, primary key), game_date (date), away_team (text), home_team (text), predicted_winner (text), win_pct (numeric 0 to 100), f5_winner (text), projected_total (numeric), created_at (timestamptz). Row-level security on; one policy lets the authenticated role select.

Commands

* Look at the scripts section of package.json first and use only scripts that exist there.
* npm run dev (local dev server), npm run build (production build check). There is no lint script.
* Run the build before pushing, and report any failure plainly.

Never

* Add a dependency without asking first.
* Edit .env or any environment variable. They are set in Vercel > ai-workshop > Settings > Environment Variables; say what is needed instead.
* Change auth configuration without saying what is changing and why.
* Create new top-level folders.
* Put keys, passwords, or tokens in a file in the repo, a prompt, or chat.
* Add a new library, service, or account without asking first.
* Use real personal data. Fake names and fake content only.
* Run, import, or re-implement the prediction model, or write to the predictions table from the website.
* Save SQL files in the repo. Database changes are given as SQL in your final message, and Niko runs them in the Supabase SQL Editor.
* Build anything listed under Backlog in roadmap.md.
* Edit roadmap.md, project-state.md, or CLAUDE.md, unless the prompt is a documentation prompt that says to.
* Merge a pull request unless the prompt explicitly says to.

Conventions

* Explain what you did in plain language, not only in code. Niko directs and reviews all work in this repo.
* Flag anything Niko would be embarrassed not to understand if asked about it at a live session.
* Keep code small and plain. Pick the simplest approach that meets the done-criteria.
* Supabase helper code lives in app/_lib/supabase/ (server.ts for server code, proxy.ts for the session refresh). Reuse these instead of creating new clients elsewhere.
* The per-request session refresh is proxy.ts at the project root. In Next.js 16 this file replaces middleware.ts. Do not add a middleware.ts.
* Every Supabase table gets row-level security (a database rule that limits who can read and change which rows). The predictions table is readable by signed-in users only, and the website never writes to it.
* When a slice needs database changes, give the SQL in separate code blocks in your final message (schema first, test data second), and explain in plain language what each row-level security policy allows and blocks.
* Like and dislike counts are visible to all signed-in users, but each person can only add, switch, or remove their own reaction.
* Games are listed by date, soonest first. The data has no start time, so games on the same date are ordered by away team name, A to Z.
* Work on your own branch. When finished, push the branch and open a draft pull request, then stop.

Current focus
See roadmap.md, work only on the slice marked ACTIVE.
