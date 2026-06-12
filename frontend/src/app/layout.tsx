import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Space_Grotesk } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

const spaceGrotesk = Space_Grotesk({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "NexusOps — Autonomous AI SRE",
  description:
    "An autonomous Site Reliability Engineer. Five LangGraph agents detect anomalies with a PyTorch LSTM, diagnose root cause with RAG, predict blast radius, and remediate Kubernetes incidents under human approval.",
  metadataBase: new URL("https://nexusops.vercel.app"),
  openGraph: {
    title: "NexusOps — Autonomous AI SRE",
    description:
      "Five agents that detect, diagnose, and heal infrastructure incidents — with a human in the loop.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable} ${spaceGrotesk.variable} dark`}
    >
      <body
        suppressHydrationWarning
        className="min-h-screen bg-ink text-foreground font-sans antialiased"
      >
        {children}
      </body>
    </html>
  );
}
