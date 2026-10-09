import { redirect } from "next/navigation";
import { createClient } from "../_lib/supabase/server";
import { logOut } from "../_lib/auth-actions";

export default async function PicksPage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  // proxy.ts already redirects logged-out visitors; this is a second check.
  if (!user) {
    redirect("/login");
  }

  return (
    <main className="container">
      <p>Signed in as {user.email}</p>
      <form action={logOut}>
        <button type="submit">Log out</button>
      </form>
    </main>
  );
}
