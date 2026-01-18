import { motion } from "framer-motion";
import { Check, X } from "lucide-react";

const comparison = [
  { traditional: "Manual scans", sentinel: "Automatic triggers" },
  { traditional: "Syntax rules", sentinel: "Semantic threat modeling" },
  { traditional: "One-off audits", sentinel: "Continuous agent" },
  { traditional: "No memory", sentinel: "Persistent context" },
  { traditional: "No tests", sentinel: "Test-driven fixes" },
];

const ComparisonTable = () => {
  return (
    <section className="section-padding relative overflow-hidden">
      <div className="absolute inset-0 bg-card/30" />

      <div className="container mx-auto px-6 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center max-w-3xl mx-auto mb-12"
        >
          <h2 className="font-display text-3xl sm:text-4xl lg:text-display-md font-bold leading-tight mb-4 text-foreground">
            Built for AI-Written Code — From Day One
          </h2>
        </motion.div>

        {/* Comparison Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.1 }}
          className="max-w-2xl mx-auto"
        >
          <div className="rounded-2xl border border-line-muted overflow-hidden bg-background">
            {/* Header */}
            <div className="grid grid-cols-2">
              <div className="px-6 py-4 border-b border-r border-line-muted bg-card/50">
                <span className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                  Traditional Tools
                </span>
              </div>
              <div className="px-6 py-4 border-b border-line-muted bg-primary/5">
                <span className="text-sm font-medium text-primary uppercase tracking-wider">
                  Sentinel
                </span>
              </div>
            </div>

            {/* Rows */}
            {comparison.map((row, index) => (
              <div
                key={index}
                className="grid grid-cols-2 border-b border-line-muted last:border-b-0"
              >
                <div className="px-6 py-4 flex items-center gap-3 border-r border-line-muted">
                  <X className="w-4 h-4 text-muted-foreground/50 flex-shrink-0" />
                  <span className="text-sm text-muted-foreground">{row.traditional}</span>
                </div>
                <div className="px-6 py-4 flex items-center gap-3 bg-primary/5">
                  <Check className="w-4 h-4 text-primary flex-shrink-0" />
                  <span className="text-sm text-foreground font-medium">{row.sentinel}</span>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Bottom callout */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2 }}
          className="text-center mt-12"
        >
          <div className="inline-flex flex-col sm:flex-row items-center gap-2 text-lg">
            <span className="text-muted-foreground">AI changed how software is written.</span>
            <span className="text-foreground font-medium">Sentinel changes how it's secured.</span>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default ComparisonTable;