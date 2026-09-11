import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "socrates",
  description: "LLM inference from first principles",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
