export default function Loader({ text = "Working..." }) {
  return (
    <div className="flex items-center gap-2 text-brand-700">
      <span className="animate-spin h-4 w-4 border-2 border-brand-500 border-t-transparent rounded-full"></span>
      <span className="text-sm">{text}</span>
    </div>
  );
}
