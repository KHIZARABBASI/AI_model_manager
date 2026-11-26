import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Model Manager",
  description: "Find and run AI models on your own computer",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
