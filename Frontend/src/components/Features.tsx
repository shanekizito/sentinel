import { motion } from "framer-motion";
import { 
  Brain,
  Search,
  Wrench,
  TestTube,
  FileText,
  Shield
} from "lucide-react";

const features = [
  {
    icon: Brain,
    title: "Security Intelligence",
    description: "Builds a living semantic map: data flows, trust boundaries, AI-agent execution paths. Updates as code changes.",
    tag: "Learning",
  },
  {
    icon: Search,
    title: "Vulnerability Detection",
    description: "Finds injections, auth bypasses, privilege escalation, and AI-specific failures. Ranked by exploitability.",
    tag: "Analysis",
  },
  {
    icon: Wrench,
    title: "Autonomous Fixes",
    description: "Minimal, targeted, reversible patches. Production-safe and compatible with future AI rewrites.",
    tag: "Auto-Patch",
  },
  {
    icon: TestTube,
    title: "Test Generation",
    description: "Writes tests that reproduce exploits, applies patches, proves they work. Guards against regressions.",
    tag: "Testing",
  },
  {
    icon: FileText,
    title: "Explanations",
    description: "Every action documented: vulnerability, exploit path, fix rationale, patterns to avoid.",
    tag: "Transparent",
  },
  {
    icon: Shield,
    title: "Continuous Protection",
    description: "Runs on pushes, PRs, deploys. Security becomes event-driven, not optional.",
    tag: "Always On",
  },
];

const Features = () => {
  return (
    <section id="features" className="py-24 relative overflow-hidden">
      <div className="absolute inset-0 bg-card/30" />
      <div className="absolute inset-0 bg-dot-pattern opacity-20" />
      
      <div className="container mx-auto px-6 relative z-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center max-w-2xl mx-auto mb-14"
        >
          <p className="text-xs font-medium text-accent uppercase tracking-wider mb-4">Capabilities</p>
          
          <h2 className="font-display text-3xl sm:text-4xl font-semibold mb-5 text-foreground tracking-tight">
            Security-First by Default
          </h2>
          
          <p className="text-muted-foreground">
            Assume the code is unsafe. Prove fixes with tests. Run continuously.
          </p>
        </motion.div>

        {/* Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5 max-w-5xl mx-auto">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.05 }}
            >
              <div className="h-full p-5 rounded-xl border border-white/5 bg-background/50 hover:border-accent/20 transition-colors group">
                <div className="flex items-center justify-between mb-4">
                  <div className="w-10 h-10 rounded-lg bg-accent/10 border border-accent/15 flex items-center justify-center group-hover:bg-accent/15 transition-colors">
                    <feature.icon className="w-5 h-5 text-accent" />
                  </div>
                  <span className="text-[10px] font-medium text-muted-foreground/50 uppercase tracking-wider">
                    {feature.tag}
                  </span>
                </div>
                
                <h3 className="font-display text-base font-semibold mb-2 text-foreground">
                  {feature.title}
                </h3>
                
                <p className="text-sm text-muted-foreground/80 leading-relaxed">
                  {feature.description}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
