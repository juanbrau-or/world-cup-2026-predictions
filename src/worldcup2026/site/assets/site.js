(() => {
  const formatUtc = (value) => {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return value || "n/a";
    }
    return date.toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    });
  };

  document.querySelectorAll("[data-local-time]").forEach((node) => {
    const value = node.getAttribute("datetime");
    if (!value) {
      return;
    }
    node.textContent = `${value} / ${formatUtc(value)}`;
    node.setAttribute("title", "UTC / hora local del navegador");
  });

  document.querySelectorAll("[data-freshness]").forEach((node) => {
    const cutoff = node.getAttribute("data-cutoff");
    const date = new Date(cutoff);
    if (!cutoff || Number.isNaN(date.getTime())) {
      node.textContent = "n/a";
      return;
    }
    const ageHours = (Date.now() - date.getTime()) / 36e5;
    if (ageHours <= 6) {
      node.textContent = "fresco";
    } else if (ageHours <= 24) {
      node.textContent = "reciente";
    } else {
      node.textContent = `desactualizado (${ageHours.toFixed(0)} h)`;
    }
  });
})();
