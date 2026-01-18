import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface AnimatedBorderProps {
  children: React.ReactNode;
  className?: string;
  borderClassName?: string;
  duration?: number;
}

const AnimatedBorder = ({
  children,
  className,
  borderClassName,
  duration = 3,
}: AnimatedBorderProps) => {
  return (
    <div className={cn("relative p-[1px] rounded-2xl", className)}>
      {/* Animated gradient border */}
      <motion.div
        className={cn(
          "absolute inset-0 rounded-2xl",
          borderClassName
        )}
        style={{
          background: "linear-gradient(90deg, hsl(var(--primary)), hsl(var(--accent)), hsl(var(--primary)))",
          backgroundSize: "200% 100%",
        }}
        animate={{
          backgroundPosition: ["0% 0%", "200% 0%"],
        }}
        transition={{
          duration,
          repeat: Infinity,
          ease: "linear",
        }}
      />
      
      {/* Content container */}
      <div className="relative z-10 bg-card rounded-2xl">
        {children}
      </div>
    </div>
  );
};

export { AnimatedBorder };
