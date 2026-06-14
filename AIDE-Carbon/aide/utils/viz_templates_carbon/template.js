const bgCol = "#F2F0E7";
const accentCol = "#fd4578";
const okCol = "#2ecc71";   // runnable node
const bugCol = "#e74c3c";  // buggy node

hljs.initHighlightingOnLoad();

const updateTargetDims = () => {
  // width is max-width of `.contentContainer` - its padding
  // return [min(windowWidth, 900 - 80), 700]
  return [windowWidth * (1 / 2), windowHeight];
};

const setCodeAndPlan = (code, plan) => {
  const codeElm = document.getElementById("code");
  if (codeElm) {
    // codeElm.innerText = code;
    codeElm.innerHTML = hljs.highlight(code, { language: "python" }).value;
  }

  const planElm = document.getElementById("plan");
  if (planElm) {
    // planElm.innerText = plan.trim();
    planElm.innerHTML = hljs.highlight(plan, { language: "plaintext" }).value;
  }
};

const fmtNum = (x) => {
  if (x === null || x === undefined) return "N/A";
  if (typeof x !== "number") return String(x);
  if (!isFinite(x)) return "N/A";
  if (Math.abs(x) >= 1000) return x.toFixed(2);
  return x.toFixed(4);
};

const renderMetrics = (treeInd) => {
  const metricsElm = document.getElementById("metrics");
  if (!metricsElm) return;

  const payloadArr = treeStructData.metrics_payload || [];
  const payload = payloadArr[treeInd];

  const schemeResults = (payload || {}).scheme_results || [];
  const delta = (payload || {}).delta_vs_baseline || null;

  // LLM score fields (arrays parallel to nodes)
  const quantScore = (treeStructData.quant_score || [])[treeInd];
  const llmScore = (treeStructData.llm_quality_score || [])[treeInd];
  const llmCA = (treeStructData.llm_carbon_awareness || [])[treeInd];
  const llmSD = (treeStructData.llm_soundness || [])[treeInd];
  const llmIV = (treeStructData.llm_innovation || [])[treeInd];
  const llmRationale = (treeStructData.llm_rationale || [])[treeInd];
  const finalScore = (quantScore != null && llmScore != null)
    ? quantScore + llmScore
    : null;

  let html = "";
  html += "<div style='font-weight:700;margin-bottom:0.5em'>Metrics</div>";

  // --- Score summary card ---
  html += "<div style='margin-bottom:0.75em;padding:0.5em;background:#1e2127;border-radius:6px'>";
  html += "<div style='font-weight:600;margin-bottom:0.3em'>Score Breakdown</div>";
  html += `<div>Final&nbsp;Score: <b style='color:#fd4578'>${fmtNum(finalScore)}</b></div>`;
  html += `<div>Quant&nbsp;Score: <span style='color:#f0c674'>${fmtNum(quantScore)}</span> <span style='font-size:0.85em;color:#888'>(0.4*time + 0.6*carbon, range ~ -50 to 50)</span></div>`;
  html += `<div>LLM&nbsp;Quality: <span style='color:#2ecc71'>${fmtNum(llmScore)}</span> <span style='font-size:0.85em;color:#888'>(scaled to ~ -10 to 10)</span></div>`;

  if (llmCA != null) {
    html += "<div style='margin-top:0.4em;font-size:0.9em;display:flex;flex-wrap:wrap;gap:0.4em 1.2em'>";
    html += `<span>Carbon&nbsp;Awareness: <b>${fmtNum(llmCA)}/10</b></span>`;
    html += `<span>Incremental&nbsp;Impr: <b>${fmtNum(llmSD)}/10</b></span>`;
    html += `<span>Extensibility: <b>${fmtNum(llmIV)}/10</b></span>`;
    html += "</div>";
  }
  if (llmRationale) {
    html += `<div style='margin-top:0.4em;font-size:0.85em;color:#abb2bf;font-style:italic;word-break:break-word;white-space:pre-wrap'>${llmRationale}</div>`;
  }
  html += "</div>";

  // --- Delta summary ---
  if (delta) {
    html += "<div style='margin-bottom:0.5em'><b>newalgo vs fifo (% improvement)</b></div>";
    html += `<div>time: ${fmtNum(delta.total_time_pct)}% | carbon: ${fmtNum(delta.total_carbon_pct)}% | jct: ${fmtNum(delta.avg_job_completion_pct)}%</div>`;
  }

  // --- Scheme results table ---
  if (schemeResults.length > 0) {
    html += "<div style='margin:0.75em 0 0.3em'><b>scheme_results</b></div>";
    html += "<div style='font-size:15px;line-height:1.4'>";
    const schemeOrder = ['decima', 'cap_decima', 'pcaps', 'fifo', 'cap_fifo', 'newalgo'];
    const schemeColors = {
      'decima': '#61afef',
      'cap_decima': '#c678dd',
      'pcaps': '#e5c07b',
      'fifo': '#abb2bf',
      'cap_fifo': '#56b6c2',
      'newalgo': '#98c379',
    };
    const ordered = schemeOrder
      .map(s => schemeResults.find(r => r.scheme === s))
      .filter(Boolean);
    const others = schemeResults.filter(r => !schemeOrder.includes(r.scheme));
    [...ordered, ...others].forEach((r) => {
      const col = schemeColors[r.scheme] || '#fd4578';
      html += `<div style='margin-bottom:3px'><b style='color:${col}'>${r.scheme}</b> &mdash; `
        + `time=<span style='color:#f0c674'>${fmtNum(r.total_time)}</span>, `
        + `carbon=<span style='color:#e06c75'>${fmtNum(r.total_carbon)}</span>, `
        + `jct=<span style='color:#56b6c2'>${fmtNum(r.avg_job_completion)}</span></div>`;
    });
    html += "</div>";
  }

  if (!payload && quantScore == null) {
    html += "<div><b>Metrics</b>: N/A</div>";
  }

  metricsElm.innerHTML = html;
};

windowResized = () => {
  resizeCanvas(...updateTargetDims());
  awaitingPostResizeOps = true;
};

const animEase = (t) => 1 - (1 - Math.min(t, 1.0)) ** 5;

// ---- global constants ----

const globalAnimSpeed = 1.1;
const scaleFactor = 0.57;

// ---- global vars ----

let globalTime = 0;
let manualSelection = false;

let currentElemInd = 0;

let treeStructData = <placeholder>

let lastClick = 0;
let firstFrameTime = undefined;

let nodes = [];
let edges = [];

let lastScrollPos = 0;

setup = () => {
  canvas = createCanvas(...updateTargetDims());
};

class Node {
  x;
  y;
  size;
  xT;
  yT;
  xB;
  yB;
  treeInd;
  color;
  relSize;
  animationStart = Number.MAX_VALUE;
  animationProgress = 0;
  isStatic = false;
  hasChildren = false;
  isRootNode = true;
  isStarred = false;
  selected = false;
  renderSize = 10;
  edges = [];
  bgCol;

  constructor(x, y, relSize, treeInd) {
    const minSize = 35;
    const maxSize = 60;

    const maxColor = 10;
    const minColor = 125;

    this.relSize = relSize;
    this.treeInd = treeInd;
    this.size = minSize + (maxSize - minSize) * relSize;
    this.color = minColor + (maxColor - minColor) * relSize;
    this.bgCol = Math.round(Math.max(this.color / 2, 0));

    this.x = x;
    this.y = y;
    this.xT = x;
    this.yT = y - this.size / 2;
    this.xB = x;
    this.yB = y + this.size / 2;

    nodes.push(this);
  }

  startAnimation = (offset = 0) => {
    if (this.animationStart == Number.MAX_VALUE)
      this.animationStart = globalTime + offset;
  };

  child = (node) => {
    let edge = new Edge(this, node);
    this.edges.push(edge);
    edges.push(edge);
    this.hasChildren = true;
    node.isRootNode = false;
    return node;
  };

  render = () => {
    if (globalTime - this.animationStart < 0) return;

    const mouseXlocalCoords = (mouseX - width / 2) / scaleFactor;
    const mouseYlocalCoords = (mouseY - height / 2) / scaleFactor;
    const isMouseOver =
      dist(mouseXlocalCoords, mouseYlocalCoords, this.x, this.y) <
      this.renderSize / 1.5;
    if (isMouseOver) cursor(HAND);
    if (isMouseOver && mouseIsPressed) {
      nodes.forEach((n) => (n.selected = false));
      this.selected = true;
      setCodeAndPlan(
        treeStructData.code[this.treeInd],
        treeStructData.plan[this.treeInd],
      );
      renderMetrics(this.treeInd);
      manualSelection = true;
    }

    this.renderSize = this.size;
    if (!this.isStatic) {
      this.animationProgress = animEase(
        (globalTime - this.animationStart) / 1000,
      );
      if (this.animationProgress >= 1) {
        this.isStatic = true;
      } else {
        this.renderSize =
          this.size *
          (0.8 +
            0.2 *
              (-3.33 * this.animationProgress ** 2 +
                4.33 * this.animationProgress));
      }
    }

    // Color nodes by status if available
    const isBuggyArr = treeStructData.is_buggy || [];
    const isBuggy = Boolean(isBuggyArr[this.treeInd]);

    fill(isBuggy ? bugCol : okCol);
    if (this.selected) {
      fill(accentCol);
    }

    noStroke();
    square(
      this.x - this.renderSize / 2,
      this.y - this.renderSize / 2,
      this.renderSize,
      10,
    );

    noStroke();
    textAlign(CENTER, CENTER);
    //textSize(this.renderSize / 2);//原先的
    //fill(255);//原先的
    // fill(lerpColor(color(accentCol), color(255), this.animationProgress))
    //text("{ }", this.x, this.y - 1);//原先的
    // DEBUG PRINT:
    // text(round(this.relSize, 2), this.x, this.y - 1)
    // text(this.treeInd, this.x, this.y + 15)


    // 用 treeInd 作为迭代号（从 1 开始显示）
    const iterNum = this.treeInd + 1;
    const iterText = String(iterNum);
    // 位数多时自动缩小字体，防止挤出方块
    const labelSize = iterText.length >= 3 ? this.renderSize * 0.30 : this.renderSize * 0.42;
    textSize(labelSize);
    fill(255);
    text(iterText, this.x, this.y - 1);

    const dotAnimThreshold = 0.85;
    if (this.isStarred && this.animationProgress >= dotAnimThreshold) {
      let dotAnimProgress =
        (this.animationProgress - dotAnimThreshold) / (1 - dotAnimThreshold);
      textSize(
        ((-3.33 * dotAnimProgress ** 2 + 4.33 * dotAnimProgress) *
          this.renderSize) /
          2,
      );
      if (this.selected) {
        fill(0);
        stroke(0);
      } else {
        fill(accentCol);
        stroke(accentCol);
      }
      strokeWeight((-(dotAnimProgress ** 2) + dotAnimProgress) * 2);
      text("*", this.x + 20, this.y - 11);
      noStroke();
    }

    if (!this.isStatic) {
      fill(bgCol);
      const progressAnimBaseSize = this.renderSize + 5;
      rect(
        this.x - progressAnimBaseSize / 2,
        this.y -
          progressAnimBaseSize / 2 +
          progressAnimBaseSize * this.animationProgress,
        progressAnimBaseSize,
        progressAnimBaseSize * (1 - this.animationProgress),
      );
    }
    if (this.animationProgress >= 0.9) {
      this.edges
        .sort((a, b) => a.color() - b.color())
        .forEach((e, i) => {
          e.startAnimation((i / this.edges.length) ** 2 * 1000);
        });
    }
  };
}

class Edge {
  nodeT;
  nodeB;
  animX = 0;
  animY = 0;
  animationStart = Number.MAX_VALUE;
  animationProgress = 0;
  isStatic = false;
  weight = 0;

  constructor(nodeT, nodeB) {
    this.nodeT = nodeT;
    this.nodeB = nodeB;
    this.weight = 2 + nodeB.relSize * 1;
  }

  color = () => this.nodeB.color;

  startAnimation = (offset = 0) => {
    if (this.animationStart == Number.MAX_VALUE)
      this.animationStart = globalTime + offset;
  };

  render = () => {
    if (globalTime - this.animationStart < 0) return;

    if (!this.isStatic) {
      this.animationProgress = animEase(
        (globalTime - this.animationStart) / 1000,
      );
      if (this.animationProgress >= 1) {
        this.isStatic = true;
        this.animX = this.nodeB.xT;
        this.animY = this.nodeB.yT;
      } else {
        this.animX = bezierPoint(
          this.nodeT.xB,
          this.nodeT.xB,
          this.nodeB.xT,
          this.nodeB.xT,
          this.animationProgress,
        );

        this.animY = bezierPoint(
          this.nodeT.yB,
          (this.nodeT.yB + this.nodeB.yT) / 2,
          (this.nodeT.yB + this.nodeB.yT) / 2,
          this.nodeB.yT,
          this.animationProgress,
        );
      }
    }
    if (this.animationProgress >= 0.97) {
      this.nodeB.startAnimation();
    }

    strokeWeight(this.weight);
    noFill();
    stroke(
      lerpColor(color(bgCol), color(accentCol), this.nodeB.relSize * 1 + 0.7),
    );
    bezier(
      this.nodeT.xB,
      this.nodeT.yB,
      this.nodeT.xB,
      (this.nodeT.yB + this.nodeB.yT) / 2,
      this.animX,
      (this.nodeT.yB + this.nodeB.yT) / 2,
      this.animX,
      this.animY,
    );
  };
}

draw = () => {
  cursor(ARROW);
  frameRate(120);
  if (!firstFrameTime && frameCount <= 1) {
    firstFrameTime = millis();
  }
  // ---- update global animation state ----
  const initialSpeedScalingEaseIO =
    (cos(min((millis() - firstFrameTime) / 8000, 1.0) * PI) + 1) / 2;
  const initialSpeedScalingEase =
    (cos(min((millis() - firstFrameTime) / 8000, 1.0) ** (1 / 2) * PI) + 1) / 2;
  const initAnimationSpeedFactor = 1.0 - 0.4 * initialSpeedScalingEaseIO;
  // update global scaling-aware clock
  globalTime += globalAnimSpeed * initAnimationSpeedFactor * deltaTime;

  if (nodes.length == 0) {
    const spacingHeight = height * 1.3;
    const spacingWidth = width * 1.3;
    treeStructData.layout.forEach((lay, index) => {
      new Node(
        spacingWidth * lay[0] - spacingWidth / 2,
        20 + spacingHeight * lay[1] - spacingHeight / 2,
        treeStructData.metrics[index],
        index,
      );
    });
    treeStructData.edges.forEach((ind) => {
      nodes[ind[0]].child(nodes[ind[1]]);
    });
    nodes.forEach((n) => {
      if (n.isRootNode) n.startAnimation();
    });
    nodes[0].selected = true;
    setCodeAndPlan(
      treeStructData.code[0],
      treeStructData.plan[0],
    )
  }

  const staticNodes = nodes.filter(
    (n) => n.isStatic || n.animationProgress >= 0.7,
  );
  if (staticNodes.length > 0) {
    const largestNode = staticNodes.reduce((prev, current) =>
      prev.relSize > current.relSize ? prev : current,
    );
    if (!manualSelection) {
      if (!largestNode.selected) {
        setCodeAndPlan(
          treeStructData.code[largestNode.treeInd],
          treeStructData.plan[largestNode.treeInd],
        );
      }
      staticNodes.forEach((node) => {
        node.selected = node === largestNode;
      });
    }
  }
  background(bgCol);
  // global animation transforms
  translate(width / 2, height / 2);
  scale(scaleFactor);

  
  // ---- fg render ----
  edges.forEach((e) => e.render());
  nodes.forEach((n) => n.render());
  
};
