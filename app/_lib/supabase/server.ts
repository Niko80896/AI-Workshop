import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";

// Supabase client for server code (pages and server actions).
// It reads and writes the sign-in cookies through Next.js.
export async function createClient() {
  const cookieStore = await cookies();

  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll();
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) =>
              cookieStore.set(name, value, options)
            );
          } catch {
            // Pages can't set cookies. That's fine: proxy.ts refreshes them.
          }
        },
      },
    }
  );
}
