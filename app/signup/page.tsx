import AuthForm from "../_lib/AuthForm";
import { signUp } from "../_lib/auth-actions";

export default function SignupPage() {
  return (
    <AuthForm
      title="Sign up"
      button="Sign up"
      action={signUp}
      otherText="Already have an account?"
      otherHref="/login"
      otherLink="Log in"
    />
  );
}
