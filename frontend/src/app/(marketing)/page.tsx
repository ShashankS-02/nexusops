import { Hero } from "@/components/marketing/hero";
import {
  StatStrip,
  Thesis,
  Lifecycle,
  Agents,
  Architecture,
  RagLoop,
  Hitl,
  Stack,
  CTA,
} from "@/components/marketing/sections";

export default function LandingPage() {
  return (
    <>
      <Hero />
      <StatStrip />
      <Thesis />
      <Lifecycle />
      <Agents />
      <Architecture />
      <RagLoop />
      <Hitl />
      <Stack />
      <CTA />
    </>
  );
}
