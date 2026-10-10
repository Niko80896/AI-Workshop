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

export default async function PicksPage() {
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

  return (
    <main className="container">
      <p>Signed in as {user.email}</p>
      <form action={logOut}>
        <button type="submit">Log out</button>
      </form>

      <h1 className="picks-title">Picks</h1>

      {error && <p className="form-error">Could not load predictions: {error.message}</p>}
      {!error && games.length === 0 && <p>No games yet.</p>}

      <ul className="games">
        {games.map((game) => (
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
