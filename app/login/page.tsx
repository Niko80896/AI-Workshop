import AuthForm from "../_lib/AuthForm";
import { logIn } from "../_lib/auth-actions";

export default function LoginPage() {
  return (
    <AuthForm
      title="Log in"
      button="Log in"
      action={logIn}
      otherText="No account yet?"
      otherHref="/signup"
      otherLink="Sign up"
    />
  );
}
