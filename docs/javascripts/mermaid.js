document$.subscribe(() => {
  mermaid.initialize({
    startOnLoad: false,
    theme: "default",
    securityLevel: "loose",
  });

  document.querySelectorAll("pre.mermaid code").forEach((codeBlock) => {
    const pre = codeBlock.parentElement;
    const container = document.createElement("div");
    container.className = "mermaid";
    container.textContent = codeBlock.textContent;
    pre.replaceWith(container);
  });

  mermaid.run({
    querySelector: ".mermaid",
  });
});
