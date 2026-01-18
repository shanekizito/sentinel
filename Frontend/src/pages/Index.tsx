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
    <div className="min-h-screen bg-background">
      <Navbar />
      <main>
        <Hero />

        {/* Social Proof / Stack */}
        <section className="py-10 border-b border-border bg-secondary/20">
          <div className="container mx-auto px-6">
            <p className="text-center text-sm text-muted-foreground mb-6">Seamlessly integrated with your modern stack</p>
            <IntegrationsGrid />
          </div>
        </section>

        <ProblemSection />

        {/* High Impact Visual: Code Diff */}
        <section className="py-24 bg-background relative overflow-hidden">
          <div className="container mx-auto px-6 relative z-10">
            <div className="text-center max-w-3xl mx-auto mb-16">
              <h2 className="font-display text-4xl font-bold mb-4 text-foreground">Don't just scan code. Fix it.</h2>
              <p className="text-muted-foreground text-lg">Sentinel understands context, imports, and type safety to generate patches you'll actually merge.</p>
            </div>
            <CodeDiffViewer />
          </div>
        </section>

        <CoreProposition />
        <FeatureShowcase />
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