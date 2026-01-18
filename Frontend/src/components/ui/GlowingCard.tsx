import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { ReactNode } from "react";

interface GlowingCardProps {
  children: ReactNode;
  className?: string;
  glowColor?: "primary" | "accent" | "violet" | "gradient";
  hoverable?: boolean;
}

const GlowingCard = ({
  children,
  className,
  glowColor = "primary",
  hoverable = true,
}: GlowingCardProps) => {
  const glowColors = {
    primary: "from-primary/20 via-transparent to-primary/10",
    accent: "from-accent/20 via-transparent to-accent/10",
    violet: "from-violet/20 via-transparent to-violet/10",
    gradient: "from-primary/20 via-accent/10 to-violet/20",
  };

  return (
    <motion.div
      className={cn(
        "relative group rounded-2xl",
        hoverable && "cursor-pointer",
        className
      )}
      whileHover={hoverable ? { y: -4, scale: 1.01 } : undefined}
      transition={{ type: "spring", stiffness: 300, damping: 20 }}
    >
      {/* Glow effect on hover */}
      <div
        className={cn(
          "absolute -inset-[1px] rounded-2xl bg-gradient-to-br opacity-0 group-hover:opacity-100 blur-xl transition-opacity duration-500",
          glowColors[glowColor]
        )}
      />
      
      {/* Border gradient */}
      <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-white/10 via-transparent to-white/5 p-[1px]">
        <div className="absolute inset-[1px] rounded-2xl bg-card" />
      </div>
      
      {/* Content */}
      <div className="relative z-10">
        {children}
      </div>
    </motion.div>
  );
};

export { GlowingCard };
