"use server";

import { redirect } from "next/navigation";
import { createClient } from "./supabase/server";

export type AuthState = { error: string } | null;

export async function signUp(_prev: AuthState, formData: FormData): Promise<AuthState> {
  const email = String(formData.get("email") ?? "");
  const password = String(formData.get("password") ?? "");

  if (password.length < 8) {
    return { error: "Password must be at least 8 characters." };
  }

  const supabase = await createClient();
  const { data, error } = await supabase.auth.signUp({ email, password });

  if (error) {
    return { error: error.message };
  }
  if (!data.session) {
    // Happens if email confirmation is turned on in Supabase.
    return { error: "Account created, but not signed in. Check that email confirmation is off in Supabase." };
  }

  redirect("/picks");
}

export async function logIn(_prev: AuthState, formData: FormData): Promise<AuthState> {
  const email = String(formData.get("email") ?? "");
  const password = String(formData.get("password") ?? "");

  const supabase = await createClient();
  const { error } = await supabase.auth.signInWithPassword({ email, password });

  if (error) {
    if (error.code === "invalid_credentials") {
      return { error: "Wrong email or password." };
    }
    return { error: error.message };
  }

  redirect("/picks");
}

export async function logOut() {
  const supabase = await createClient();
  await supabase.auth.signOut();
  redirect("/login");
}
