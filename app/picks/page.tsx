import Link from "next/link";
import { redirect } from "next/navigation";
import { createClient } from "../_lib/supabase/server";
import { logOut } from "../_lib/auth-actions";

type Prediction = {
  id: number;
  game_date: string;
  away_team: string;
  home_team: string;
  predicted_winner: string;
  win_pct: number;
  f5_winner: string;
  projected_total: number;
};

type Tab = "past" | "upcoming" | "future";

const TABS: { key: Tab; label: string; empty: string }[] = [
  { key: "past", label: "Past", empty: "No past games." },
  { key: "upcoming", label: "Upcoming", empty: "No games today." },
  { key: "future", label: "Future", empty: "No future games." },
];

// Today's date in Hawaii as "YYYY-MM-DD", the same shape as game_date.
// The "en-CA" format writes dates as year-month-day.
function hawaiiToday(): string {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "Pacific/Honolulu",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());
}

export default async function PicksPage({
  searchParams,
}: {
  searchParams: Promise<{ tab?: string }>;
}) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  // proxy.ts already redirects logged-out visitors; this is a second check.
  if (!user) {
    redirect("/login");
  }

  // Soonest date first; same-date games ordered by away team, A to Z.
  const { data, error } = await supabase
    .from("predictions")
    .select("id, game_date, away_team, home_team, predicted_winner, win_pct, f5_winner, projected_total")
    .order("game_date", { ascending: true })
    .order("away_team", { ascending: true });

  const games = (data ?? []) as Prediction[];

  // Each game lands in exactly one tab by comparing its date to Hawaii's today.
  const today = hawaiiToday();
  const byTab: Record<Tab, Prediction[]> = {
    past: games.filter((g) => g.game_date < today),
    upcoming: games.filter((g) => g.game_date === today),
    future: games.filter((g) => g.game_date > today),
  };

  // The clicked tab comes from the address (?tab=past). With none,
  // open Upcoming, or Future if nothing is dated today.
  const { tab: requested } = await searchParams;
  const tab: Tab =
    requested === "past" || requested === "upcoming" || requested === "future"
      ? requested
      : byTab.upcoming.length > 0
        ? "upcoming"
        : "future";
  const shown = byTab[tab];

  return (
    <main className="container">
      <p>Signed in as {user.email}</p>
      <form action={logOut}>
        <button type="submit">Log out</button>
      </form>

      <h1 className="picks-title">Picks</h1>

      {error && <p className="form-error">Could not load predictions: {error.message}</p>}

      <nav className="tabs">
        {TABS.map((t) => (
          <Link
            key={t.key}
            href={`/picks?tab=${t.key}`}
            className={t.key === tab ? "tab tab-selected" : "tab"}
            aria-current={t.key === tab ? "page" : undefined}
          >
            {t.label}
          </Link>
        ))}
      </nav>

      {!error && shown.length === 0 && <p>{TABS.find((t) => t.key === tab)!.empty}</p>}

      <ul className="games">
        {shown.map((game) => (
          <li key={game.id} className="game">
            <h2>
              {game.away_team} at {game.home_team}
            </h2>
            <dl>
              <dt>Date</dt>
              <dd>{game.game_date}</dd>
              <dt>Away</dt>
              <dd>{game.away_team}</dd>
              <dt>Home</dt>
              <dd>{game.home_team}</dd>
              <dt>Predicted winner</dt>
              <dd>
                {game.predicted_winner} {Number(game.win_pct).toFixed(1)}%
              </dd>
              <dt>F5 winner</dt>
              <dd>{game.f5_winner}</dd>
              <dt>Projected total runs</dt>
              <dd>{Number(game.projected_total).toFixed(1)}</dd>
            </dl>
          </li>
        ))}
      </ul>
    </main>
  );
}
