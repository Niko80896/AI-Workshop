"use client";

import Link from "next/link";
import { useActionState } from "react";
import type { AuthState } from "./auth-actions";

type Props = {
  title: string;
  button: string;
  action: (prev: AuthState, formData: FormData) => Promise<AuthState>;
  otherText: string;
  otherHref: string;
  otherLink: string;
};

// The shared email-and-password form used by /login and /signup.
export default function AuthForm({ title, button, action, otherText, otherHref, otherLink }: Props) {
  const [state, formAction, pending] = useActionState(action, null);

  return (
    <main className="container">
      <h1>{title}</h1>
      <form action={formAction} className="auth-form">
        <label>
          Email
          <input name="email" type="email" required autoComplete="email" />
        </label>
        <label>
          Password
          <input name="password" type="password" required minLength={8} autoComplete="current-password" />
        </label>
        {state?.error && <p className="form-error" role="alert">{state.error}</p>}
        <button type="submit" disabled={pending}>
          {button}
        </button>
      </form>
      <p>
        {otherText} <Link href={otherHref}>{otherLink}</Link>
      </p>
    </main>
  );
}
