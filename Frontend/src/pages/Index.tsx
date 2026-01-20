import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import ProblemSection from "@/components/ProblemSection";
import CoreProposition from "@/components/CoreProposition";
// FeatureShowcase moved to dedicated pages? No, keeping FeatureShowcase on Home.
import { FeatureShowcase } from "@/components/FeatureShowcase";
import ComparisonTable from "@/components/ComparisonTable";
import TrustSection from "@/components/TrustSection";
import UseCasesSection from "@/components/UseCasesSection";
import CTA from "@/components/CTA";
import Footer from "@/components/Footer";
import IntegrationsGrid from "@/components/IntegrationsGrid";
import CodeDiffViewer from "@/components/CodeDiffViewer";

const Index = () => {
  return (
    <div className="min-h-screen bg-background text-foreground font-sans">
      <Navbar />
      <main>
        <Hero />

        {/* Social Proof / Stack - Lighter Treatment */}
        <section className="py-12 border-b border-border/40 bg-[#FAfaf9]">
          <div className="container mx-auto px-6">
            <div className="flex flex-col md:flex-row items-center justify-center gap-6 md:gap-12 opacity-70 grayscale hover:grayscale-0 transition-all duration-500">
              <IntegrationsGrid />
            </div>
          </div>
        </section>

        <ProblemSection />

        <CoreProposition />
        <FeatureShowcase />

        {/* Code Diff Section */}
        <section className="py-32 bg-white relative overflow-hidden">
          <div className="container mx-auto px-6">
            <div className="text-center mb-16">
              <h2 className="font-display text-4xl font-bold mb-4">See the difference</h2>
              <p className="text-muted-foreground text-lg">Sentinel doesn't just flag lines. It understands logic.</p>
            </div>
            <div className="max-w-5xl mx-auto shadow-2xl rounded-xl overflow-hidden border border-border">
              <CodeDiffViewer />
            </div>
          </div>
        </section>

        <ComparisonTable />
        <TrustSection />
        <UseCasesSection />
        <CTA />
      </main>
      <Footer />
    </div>
  );
};

export default Index;