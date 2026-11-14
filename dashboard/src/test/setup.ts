import "@testing-library/jest-dom/vitest";

class ResizeObserverStub implements ResizeObserver {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}

globalThis.ResizeObserver = ResizeObserverStub;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
(HTMLCanvasElement.prototype as any).getContext = function () {
  return {
    canvas: this,
    createImageData: () => ({ data: new Uint8ClampedArray(4) }),
    putImageData: () => {},
    clearRect: () => {},
    fillRect: () => {},
    beginPath: () => {},
    moveTo: () => {},
    lineTo: () => {},
    closePath: () => {},
    stroke: () => {},
    fill: () => {},
    arc: () => {},
    fillText: () => {},
    strokeRect: () => {},
    setLineDash: () => {},
    createLinearGradient: () => ({ addColorStop: () => {} }),
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
    font: "",
  };
};