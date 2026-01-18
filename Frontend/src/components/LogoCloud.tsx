import { motion } from "framer-motion";

const LogoCloud = () => {
  const tools = [
    "Cursor",
    "Copilot", 
    "Claude",
    "Lovable",
    "Replit",
    "v0",
  ];

  return (
    <section className="py-12 border-y border-white/5 bg-card/30">
      <div className="container mx-auto px-6">
        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="text-center text-xs text-muted-foreground/60 uppercase tracking-wider mb-6"
        >
          Built for teams using AI-powered development
        </motion.p>
        
        <div className="flex flex-wrap items-center justify-center gap-x-10 gap-y-4">
          {tools.map((name, i) => (
            <motion.span
              key={name}
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.05 }}
              className="text-base font-medium text-muted-foreground/40 hover:text-muted-foreground/60 transition-colors cursor-default"
            >
              {name}
            </motion.span>
          ))}
        </div>
      </div>
    </section>
  );
};

export default LogoCloud;
