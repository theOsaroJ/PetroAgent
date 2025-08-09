export default function FancyCard({ children, title, subtitle, className="" }) {
  return (
    <div className={`rounded-2xl shadow-lg bg-white/80 backdrop-blur-sm border border-slate-200 ${className}`}>
      {title && (
        <div className="px-6 pt-6">
          <h2 className="text-xl font-semibold">{title}</h2>
          {subtitle && <p className="text-slate-500 text-sm mt-1">{subtitle}</p>}
        </div>
      )}
      <div className="p-6">{children}</div>
    </div>
  );
}
