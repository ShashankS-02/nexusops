import { MarketingNav } from "@/components/marketing/marketing-nav";
import { MarketingFooter } from "@/components/marketing/marketing-footer";

export default function MarketingLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <div className="grain relative min-h-screen bg-ink text-foreground antialiased overflow-x-hidden">
      <MarketingNav />
      <main className="relative z-[2]">{children}</main>
      <MarketingFooter />
    </div>
  );
}
