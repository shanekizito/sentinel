import { motion } from "framer-motion";

const useCases = [
  {
    type: "Solo Founder",
    quote: "“I can’t afford to be wrong.”"
  },
  {
    type: "AI SaaS Team",
    quote: "“Sentinel runs on every PR.”"
  },
  {
    type: "Fintech / Web3",
    quote: "“Security isn’t optional.”"
  },
  {
    type: "Open Source Maintainer",
    quote: "“Regressions don’t slip back in.”"
  }
];

const UseCasesSection = () => {
  return (
    <section className="section-padding relative overflow-hidden">
      <div className="container mx-auto px-6 relative z-10">
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6 max-w-5xl mx-auto">
          {useCases.map((useCase, index) => (
            <motion.div
              key={useCase.type}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="p-8 rounded-2xl border border-line-muted bg-card/10 hover:bg-card/40 transition-colors text-center group"
            >
              <div className="text-sm font-mono text-muted-foreground uppercase tracking-wider mb-4 group-hover:text-primary transition-colors">
                {useCase.type}
              </div>
              <div className="font-display text-xl font-medium text-foreground">
                {useCase.quote}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default UseCasesSection;