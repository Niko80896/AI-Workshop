import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Niko Banto",
  description: "a senior at UH Manoa studying business management",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
