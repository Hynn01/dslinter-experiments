#!/usr/bin/env python
# coding: utf-8

# In[ ]:


*(eslint-disable, no-unused-vars, */)
const chartContent = [
  '```chart',
  ',category1,category2',
  'Jan,21,23',
  'Feb,31,17',
  '',
  'type: column',
  'title: Monthly Revenue',
  'x.title: Amount',
  'y.title: Month',
  'y.min: 1',
  'y.max: 40',
  'y.suffix: $',
  '```'
].join('\n');
 
const codeContent = [
  '```js',
  `console.log('foo')`,
  '```',
  '```javascript',
  `console.log('bar')`,
  '```',
  '```html',
  '<div id="editor"><span>baz</span></div>',
  '```',
  '```wrong',
  '[1 2 3]',
  '```',
  '```clojure',
  '[1 2 3]',
  '```'
].join('\n');
 
const tableContent = ['| @cols=2:merged |', '| --- | --- |', '| table | table2 |'].join('\n');
 
const umlContent = [
  '```uml',
  'partition Conductor {',
  '  (*) --> "Climbs on Platform"',
  '  --> === S1 ===',
  '  --> Bows',
  '}',
  '',
  'partition Audience #LightSkyBlue {',
  '  === S1 === --> Applauds',
  '}',
  '',
  'partition Conductor {',
  '  Bows --> === S2 ===',
  '  --> WavesArmes',
  '  Applauds --> === S2 ===',
  '}',
  '',
  'partition Orchestra #CCCCEE {',
  '  WavesArmes --> Introduction',
  '  --> "Play music"',
  '}',
  '```'
].join('\n');
 
const allPluginsContent = [chartContent, codeContent, tableContent, umlContent].join('\n');
 


# In[ ]:


<!DOCTYPE html>
<html>
  <head lang="en">
    <meta charset="UTF-8" />
    <title>6. Editor with Chart Plugin</title>
    <link rel="stylesheet" href="./css/tuidoc-example-style.css" />
    <!-- Editor's Dependencies -->
    <link
      rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.48.4/codemirror.css"
    (>)
    <!-- Editor -->
    <link rel="stylesheet" href="../dist/cdn/toastui-editor.css" />
    <!-- Editor's Plugin -->
    <link rel="stylesheet" href="https://uicdn.toast.com/tui.chart/v3.7.0/tui-chart.css" />
  </head>
  <body>
    <div class="tui-doc-description">
      <strong
        >The example code can be slower than your environment because the code is transpiled by
        babel-standalone in runtime.</strong
      >
      <br />
      You can see the tutorial
      <a
        href="https://github.com/nhn/tui.editor/blob/master/apps/editor/docs/plugins.md"
        target="_blank"
        >here</a
      >.
    </div>
    <div class="code-html tui-doc-contents">
      <!-- Editor -->
      <h2>Editor</h2>
      <div id="editor"></div>
      <!-- Viewer Using Editor -->
      <h2>Viewer</h2>
      <div id="viewer"></div>
    </div>
    <!-- Added to check demo page in Internet Explorer -->
    <script src="https://unpkg.com/babel-standalone@6.26.0/babel.min.js"></script>
    <script src="./data/md-plugins.js"></script>
    <!-- Editor -->
    <script src="../dist/cdn/toastui-editor-all.js"></script>
    <!-- Editor's Plugin -->
    <script src="https://uicdn.toast.com/editor-plugin-chart/1.0.0/toastui-editor-plugin-chart.min.js"></script>
    <script type="text/babel" class="code-js">
      const { Editor } = toastui;
      const { chart } = Editor.plugin;
 
      const chartOptions = {
        minWidth: 100,
        maxWidth: 600,
        minHeight: 100,
        maxHeight: 300
      };
 
      const editor = new Editor({
        el: document.querySelector('#editor'),
        previewStyle: 'vertical',
        height: '500px',
        initialValue: chartContent,
        plugins: [chart, chartOptions]
      });
 
      const viewer = Editor.factory({
        el: document.querySelector('#viewer'),
        viewer: true,
        height: '500px',
        initialValue: chartContent,
        plugins: [[chart, chartOptions]]
      });
    </script>
  </body>
</html>
 


# In[ ]:


**()
 * @fileoverview configs file for bundling
 * @author NHN FE Development Lab <dl_javascript@nhn.com>
 */
const path = require('path');
const webpack = require('webpack');
const pkg = require('./package.json');
 
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const OptimizeCSSAssetsPlugin = require('optimize-css-assets-webpack-plugin');
const TerserPlugin = require('terser-webpack-plugin');
const { BundleAnalyzerPlugin } = require('webpack-bundle-analyzer');
const FileManagerPlugin = require('filemanager-webpack-plugin');
 
const ENTRY_EDITOR = './src/js/index.js';
const ENTRY_VIEWER = './src/js/indexViewer.js';
 
const isDevelopAll = process.argv.indexOf('--all') >= 0;
const isDevelopViewer = process.argv.indexOf('--viewer') >= 0;
const isProduction = process.argv.indexOf('--mode=production') >= 0;
const minify = process.argv.indexOf('--minify') >= 0;
 
const defaultConfigs = Array(isProduction ? 2 : 1)
  .fill(0)
  .map(() => {
    return {
      mode: isProduction ? 'production' : 'development',
      cache: false,
      output: {
        library: ['toastui', 'Editor'],
        libraryTarget: 'umd',
        libraryExport: 'default',
        path: path.resolve(__dirname, minify ? 'dist/cdn' : 'dist'),
        filename: `toastui-[name]${minify ? '.min' : ''}.js`
      },
      module: {
        rules: [
          {
            test: /\.js$/,
            exclude: /node_modules|dist|build/,
            loader: 'eslint-loader',
            enforce: 'pre',
            options: {
              configFile: './.eslintrc.js',
              failOnWarning: false,
              failOnError: false
            }
          },
          {
            test: /\.js$/,
            exclude: /node_modules|dist|build/,
            loader: 'babel-loader?cacheDirectory',
            options: {
              envName: isProduction ? 'production' : 'development',
              rootMode: 'upward'
            }
          },
          {
            test: /\.css$/,
            use: [MiniCssExtractPlugin.loader, 'css-loader']
          },
          {
            test: /\.png$/i,
            use: 'url-loader'
          }
        ]
      },
      plugins: [
        new MiniCssExtractPlugin({
          moduleFilename: ({ name }) =>
            `toastui-${name.replace('-all', '')}${minify ? '.min' : ''}.css`
        }),
        new webpack.BannerPlugin({
          banner: [
            pkg.name,
            `@version ${pkg.version} | ${new Date().toDateString()}`,
            `@author ${pkg.author}`,
            `@license ${pkg.license}`
          ].join('\n'),
          raw: false,
          entryOnly: true
        })
      ],
      externals: [
        {
          codemirror: {
            commonjs: 'codemirror',
            commonjs2: 'codemirror',
            amd: 'codemirror',
            root: ['CodeMirror']
          }
        }
      ],
      optimization: {
        minimize: false
      },
      performance: {
        hints: false
      }
    };
  });
 
function addFileManagerPlugin(config) {
  // When an entry option's value is set to a CSS file,
  (/, empty, JavaScript, files, are, created., (e.g., toastui-editor-only.js))
  (/, These, files, are, unnecessary,, so, use, the, FileManager, plugin, to, delete, them.)
  const options = minify
    get_ipython().run_line_magic('pinfo', '')
        {
          delete: [
            './dist/cdn/toastui-editor-only.min.js',
            './dist/cdn/toastui-editor-old.min.js',
            './dist/cdn/toastui-editor-viewer-old.min.js'
          ]
        }
      ]
    : [
        {
          delete: [
            './dist/toastui-editor-only.js',
            './dist/toastui-editor-old.js',
            './dist/toastui-editor-viewer-old.js'
          ]
        },
        { copy: [{ source: './dist/*.{js,css}', destination: './dist/cdn' }] }
      ];
 
  config.plugins.push(new FileManagerPlugin({ onEnd: options }));
}
 
function addMinifyPlugin(config) {
  config.optimization = {
    minimizer: [
      new TerserPlugin({
        cache: true,
        parallel: true,
        sourceMap: false,
        extractComments: false
      }),
      new OptimizeCSSAssetsPlugin()
    ]
  };
}
 
function addAnalyzerPlugin(config, type) {
  config.plugins.push(
    new BundleAnalyzerPlugin({
      analyzerMode: 'static',
      reportFilename: `../../report/webpack/stats-${pkg.version}-${type}.html`
    })
  );
}
 
function setDevelopConfig(config) {
  if (isDevelopAll) {
    // check in examples
    config.entry = { 'editor-all': ENTRY_EDITOR };
    config.output.publicPath = 'dist/cdn';
    config.externals = [];
  } else if (isDevelopViewer) {
    // check in examples
    config.entry = { 'editor-viewer': ENTRY_VIEWER };
    config.output.publicPath = 'dist/cdn';
  } else {
    // check in demo
    config.module.rules = config.module.rules.slice(1);
    config.entry = { editor: ENTRY_EDITOR };
    config.output.publicPath = 'dist/';
  }
 
  config.devtool = 'inline-source-map';
  config.devServer = {
    inline: true,
    host: '0.0.0.0',
    port: 8080,
    disableHostCheck: true
  };
}
 
function setProductionConfig(config) {
  config.entry = {
    editor: ENTRY_EDITOR,
    'editor-viewer': ENTRY_VIEWER,
    'editor-only': './src/js/indexEditorOnlyStyle.js',
    // legacy styles
    'editor-old': './src/js/indexOldStyle.js',
    'editor-viewer-old': './src/css/old/contents.css'
  };
 
  addFileManagerPlugin(config);
 
  if (minify) {
    addMinifyPlugin(config);
    addAnalyzerPlugin(config, 'normal');
  }
}
 
function setProductionConfigForAll(config) {
  config.entry = { 'editor-all': ENTRY_EDITOR };
  config.output.path = path.resolve(__dirname, 'dist/cdn');
  config.externals = [];
 
  if (minify) {
    addMinifyPlugin(config);
    addAnalyzerPlugin(config, 'all');
  }
}
 
if (isProduction) {
  setProductionConfig(defaultConfigs[0]);
  setProductionConfigForAll(defaultConfigs[1]);
} else {
  setDevelopConfig(defaultConfigs[0]);
}
 
module.exports = defaultConfigs;
 


# In[ ]:


(/, Type, definitions, for, TOAST, UI, Editor, v2.1.0)
(/, TypeScript, Version:, 3.2.2)
 
(//, <reference, types="codemirror", />)
 
declare namespace toastui {
  type SquireExt = any;
  type HandlerFunc = (...args: any[]) => void;
  type ReplacerFunc = (inputString: string) => string;
  type CodeMirrorType = CodeMirror.EditorFromTextArea;
  type CommandManagerExecFunc = (name: string, ...args: any[]) => any;
  type PopupTableUtils = LayerPopup;
  type AddImageBlobHook = (fileOrBlob: File | Blob, callback: Function, source: string) => void;
  type Plugin = (editor: Editor | Viewer, options: any) => void;
  type PreviewStyle = 'tab' | 'vertical';
  type CustomHTMLSanitizer = (content: string) => string | DocumentFragment;
  type LinkAttribute = Partial<{
    rel: string;
    target: string;
    contenteditable: boolean | 'true' | 'false';
    hreflang: string;
    type: string;
  }>;
  type AutolinkParser = (
    content: string
  ) => {
    url: string;
    text: string;
    range: [number, number];
  }[];
  type ExtendedAutolinks = boolean | AutolinkParser;
  type Sanitizer = (content: string) => string | DocumentFragment;
 
  // @TODO: change toastMark type definition to @toast-ui/toastmark type file through importing
  // Toastmark custom renderer type
  type BlockNodeType =
    | 'document'
    | 'list'
    | 'blockQuote'
    | 'item'
    | 'heading'
    | 'thematicBreak'
    | 'paragraph'
    | 'codeBlock'
    | 'htmlBlock'
    | 'table'
    | 'tableHead'
    | 'tableBody'
    | 'tableRow'
    | 'tableCell'
    | 'tableDelimRow'
    | 'tableDelimCell'
    | 'refDef';
 
  type InlineNodeType =
    | 'code'
    | 'text'
    | 'emph'
    | 'strong'
    | 'strike'
    | 'link'
    | 'image'
    | 'htmlInline'
    | 'linebreak'
    | 'softbreak';
 
  type NodeType = BlockNodeType | InlineNodeType;
  type SourcePos = [[number, number], [number, number]];
 
  interface NodeWalker {
    current: MdNode | null;
    root: MdNode;
    entering: boolean;
 
    next(): { entering: boolean; node: MdNode } | null;
    resumeAt(node: MdNode, entering: boolean): void;
  }
 
  interface MdNode {
    type: NodeType;
    id: number;
    parent: MdNode | null;
    prev: MdNode | null;
    next: MdNode | null;
    sourcepos?: SourcePos;
    firstChild: MdNode | null;
    lastChild: MdNode | null;
    literal: string | null;
 
    isContainer(): boolean;
    unlink(): void;
    replaceWith(node: MdNode): void;
    insertAfter(node: MdNode): void;
    insertBefore(node: MdNode): void;
    appendChild(child: MdNode): void;
    prependChild(child: MdNode): void;
    walker(): NodeWalker;
  }
 
  interface TagToken {
    tagName: string;
    outerNewLine?: boolean;
    innerNewLine?: boolean;
  }
 
  interface OpenTagToken extends TagToken {
    type: 'openTag';
    classNames?: string[];
    attributes?: Record<string, string>;
    selfClose?: boolean;
  }
 
  interface CloseTagToken extends TagToken {
    type: 'closeTag';
  }
 
  interface TextToken {
    type: 'text';
    content: string;
  }
 
  interface RawHTMLToken {
    type: 'html';
    content: string;
    outerNewLine?: boolean;
  }
 
  type HTMLToken = OpenTagToken | CloseTagToken | TextToken | RawHTMLToken;
 
  interface ContextOptions {
    gfm: boolean;
    softbreak: string;
    nodeId: boolean;
    tagFilter: boolean;
    convertors?: CustomHTMLRendererMap;
  }
 
  interface Context {
    entering: boolean;
    leaf: boolean;
    options: Omit<ContextOptions, 'gfm' | 'convertors'>;
    getChildrenText: (node: MdNode) => string;
    skipChildren: () => void;
    origin?: () => ReturnType<CustomHTMLRenderer>;
  }
 
  export type CustomHTMLRenderer = (node: MdNode, context: Context) => HTMLToken | HTMLToken[] | null;
 
  type CustomHTMLRendererMap = Partial<Record<NodeType, CustomHTMLRenderer>>;
  (/, Toastmark, custom, renderer, type, end)
  interface SelectionRange {
    from: {
      row: number;
      cell: number;
    };
    to: {
      row: number;
      cell: number;
    };
  }
 
  interface ToolbarState {
    strong: boolean;
    emph: boolean;
    strike: boolean;
    code: boolean;
    codeBlock: boolean;
    blockQuote: boolean;
    table: boolean;
    heading: boolean;
    list: boolean;
    orderedList: boolean;
    taskList: boolean;
  }
 
  type WysiwygToolbarState = ToolbarState & {
    source: 'wysiwyg';
  };
 
  type MarkdownToolbarState = ToolbarState & {
    thematicBreak: boolean;
    source: 'markdown';
  };
 
  type SourceType = 'wysiwyg' | 'markdown';
 
  interface EventMap {
    load?: (param: Editor) => void;
    change?: (param: { source: SourceType | 'viewer'; data: MouseEvent }) => void;
    stateChange?: (param: MarkdownToolbarState | WysiwygToolbarState) => void;
    focus?: (param: { source: SourceType }) => void;
    blur?: (param: { source: SourceType }) => void;
  }
 
  interface ViewerHookMap {
    previewBeforeHook?: (html: string) => void | string;
  }
 
  type EditorHookMap = ViewerHookMap & {
    addImageBlobHook?: (
      blob: Blob | File,
      callback: (url: string, altText: string) => void
    ) => void;
  };
 
  interface ToMarkOptions {
    gfm?: boolean;
    renderer?: any;
  }
 
  export interface Convertor {
    initHtmlSanitizer(sanitizer: Sanitizer): void;
    toHTML(makrdown: string): string;
    toHTMLWithCodeHighlight(markdown: string): string;
    toMarkdown(html: string, toMarkdownOptions: ToMarkOptions): string;
  }
 
  export interface ConvertorClass {
    new (em: EventManager, options: ConvertorOptions): Convertor;
  }
 
  export interface ConvertorOptions {
    linkAttribute: LinkAttribute;
    customHTMLRenderer: CustomHTMLRenderer;
    extendedAutolinks: boolean | AutolinkParser;
    referenceDefinition: boolean;
  }
 
  export interface EditorOptions {
    el: HTMLElement;
    height?: string;
    minHeight?: string;
    initialValue?: string;
    previewStyle?: PreviewStyle;
    initialEditType?: string;
    events?: EventMap;
    hooks?: EditorHookMap;
    language?: string;
    useCommandShortcut?: boolean;
    useDefaultHTMLSanitizer?: boolean;
    usageStatistics?: boolean;
    toolbarItems?: (string | ToolbarButton)[];
    hideModeSwitch?: boolean;
    plugins?: Plugin[];
    extendedAutolinks?: ExtendedAutolinks;
    customConvertor?: ConvertorClass;
    placeholder?: string;
    linkAttribute?: LinkAttribute;
    customHTMLRenderer?: CustomHTMLRenderer;
    referenceDefinition?: boolean;
    customHTMLSanitizer?: CustomHTMLSanitizer;
    previewHighlight?: boolean;
  }
 
  export interface ViewerOptions {
    el: HTMLElement;
    initialValue?: string;
    events?: EventMap;
    hooks?: ViewerHookMap;
    plugins?: Plugin[];
    useDefaultHTMLSanitizer?: boolean;
    extendedAutolinks?: ExtendedAutolinks;
    customConvertor?: ConvertorClass;
    linkAttribute?: LinkAttribute;
    customHTMLRenderer?: CustomHTMLRenderer;
    referenceDefinition?: boolean;
    customHTMLSanitizer?: CustomHTMLSanitizer;
  }
 
  interface MarkdownEditorOptions {
    height?: string;
  }
 
  interface WysiwygEditorOptions {
    useDefaultHTMLSanitizer?: boolean;
    linkAttribute?: LinkAttribute;
  }
 
  interface LanguageData {
    [propType: string]: string;
  }
 
  interface ToolbarButton {
    type: string;
    options: ButtonOptions;
  }
 
  interface ButtonOptions {
    el?: HTMLElement;
    className?: string;
    command?: string;
    event?: string;
    text?: string;
    tooltip?: string;
    style?: string;
    state?: string;
  }
 
  class UIController {
    public tagName: string;
 
    public className: string;
 
    public el: HTMLElement;
 
    public on(aType: string | object, aFn: (...args: any[]) => void): void;
 
    public off(type: string, fn: (...args: any[]) => void): void;
 
    public remove(): void;
 
    public trigger(eventTypeEvent: string, eventData?: any): void;
 
    public destroy(): void;
  }
 
  class ToolbarItem extends UIController {
    public static name: string;
 
    public static className: string;
 
    public getName(): string;
  }
 
  interface CommandType {
    MD: 0;
    WW: 1;
    GB: 2;
  }
 
  interface CommandProps {
    name: string;
    type: number;
  }
 
  class Command {
    public static TYPE: CommandType;
 
    public static factory(typeStr: string, props: CommandProps): Command;
 
    constructor(name: string, type: number, keyMap?: string[]);
 
    public getName(): string;
 
    public getType(): number;
 
    public isGlobalType(): boolean;
 
    public isMDType(): boolean;
 
    public isWWType(): boolean;
 
    public setKeyMap(win: string, mac: string): void;
  }
 
  interface LayerPopupOptions {
    openerCssQuery?: string[];
    closerCssQuery?: string[];
    el: HTMLElement;
    content?: HTMLElement | string;
    textContent?: string;
    title: string;
    header?: boolean;
    target?: HTMLElement;
    modal: boolean;
    headerButtons?: string;
  }
 
  interface LayerPopup extends UIController {
    setContent(content: HTMLElement): void;
    setTitle(title: string): void;
    getTitleElement(): HTMLElement;
    hide(): void;
    show(): void;
    isShow(): boolean;
    remove(): void;
    setFitToWindow(fit: boolean): void;
    isFitToWindow(): boolean;
    toggleFitToWindow(): boolean;
  }
 
  interface ModeSwitchType {
    MARKDOWN: 'markdown';
    WYSIWYG: 'wysiwyg';
  }
 
  interface ModeSwitch extends UIController {
    TYPE: ModeSwitchType;
    isShown(): boolean;
    show(): void;
    hide(): void;
  }
 
  class Toolbar extends UIController {
    public disableAllButton(): void;
 
    public enableAllButton(): void;
 
    public getItems(): ToolbarItem[];
 
    public getItem(index: number): ToolbarItem;
 
    public setItems(items: ToolbarItem[]): void;
 
    public addItem(item: ToolbarItem | ToolbarButton | string): void;
 
    public insertItem(index: number, item: ToolbarItem | ToolbarButton | string): void;
 
    public indexOfItem(item: ToolbarItem): number;
 
    public removeItem(item: ToolbarItem | number, destroy?: boolean): ToolbarItem | undefined;
 
    public removeAllItems(): void;
  }
 
  interface UI {
    createPopup(options: LayerPopupOptions): LayerPopup;
    getEditorHeight(): number;
    getEditorSectionHeight(): number;
    getModeSwitch(): ModeSwitch;
    getPopupTableUtils(): PopupTableUtils;
    getToolbar(): Toolbar;
    hide(): void;
    remove(): void;
    setToolbar(toolbar: Toolbar): void;
    show(): void;
  }
 
  interface CommandManagerOptions {
    useCommandShortcut?: boolean;
  }
 
  interface CommandPropsOptions {
    name: string;
    keyMap?: string[];
    exec?: CommandManagerExecFunc;
  }
 
  class CommandManager {
    public static command(type: string, props: CommandPropsOptions): Command;
 
    constructor(base: Editor, options?: CommandManagerOptions);
 
    public addCommand(command: Command): Command;
 
    public exec(name: string, ...args: any[]): any;
  }
 
  class CodeBlockManager {
    public createCodeBlockHtml(language: string, codeText: string): string;
 
    public getReplacer(language: string): ReplacerFunc;
 
    public setReplacer(language: string, replacer: ReplacerFunc): void;
  }
 
  interface RangeType {
    start: {
      line: number;
      ch: number;
    };
    end: {
      line: number;
      ch: number;
    };
  }
 
  interface MdTextObject {
    setRange(range: RangeType): void;
    setEndBeforeRange(range: RangeType): void;
    expandStartOffset(): void;
    expandEndOffset(): void;
    getTextContent(): RangeType;
    replaceContent(content: string): void;
    deleteContent(): void;
    peekStartBeforeOffset(offset: number): RangeType;
  }
 
  interface WwTextObject {
    deleteContent(): void;
    expandEndOffset(): void;
    expandStartOffset(): void;
    getTextContent(): string;
    peekStartBeforeOffset(offset: number): string;
    replaceContent(content: string): void;
    setEndBeforeRange(range: Range): void;
    setRange(range: Range): void;
  }
 
  interface FindOffsetNodeInfo {
    container: Node;
    offsetInContainer: number;
    offset: number;
  }
 
  interface NodeInfo {
    id?: string;
    tagName: string;
    className?: string;
  }
 
  class WwCodeBlockManager {
    constructor(wwe: WysiwygEditor);
 
    public destroy(): void;
 
    public convertNodesToText(nodes: Node[]): string;
 
    public isInCodeBlock(range: Range): boolean;
 
    public prepareToPasteOnCodeblock(nodes: Node[]): DocumentFragment;
 
    public modifyCodeBlockForWysiwyg(node: HTMLElement): void;
  }
 
  class WwTableManager {
    constructor(wwe: WysiwygEditor);
 
    public destroy(): void;
 
    public getTableIDClassName(): string;
 
    public isInTable(range: Range): boolean;
 
    public isNonTextDeleting(range: Range): boolean;
 
    public isTableOrSubTableElement(pastingNodeName: string): boolean;
 
    public pasteClipboardData(clipboardTable: Node): boolean;
 
    public prepareToTableCellStuffing(
      trs: HTMLElement
    ): { maximumCellLength: number; needTableCellStuffingAid: boolean };
 
    public resetLastCellNode(): void;
 
    public setLastCellNode(node: HTMLElement): void;
 
    public tableCellAppendAidForTableElement(node: HTMLElement): void;
 
    public updateTableHtmlOfClipboardIfNeed(clipboardContainer: HTMLElement): void;
 
    public wrapDanglingTableCellsIntoTrIfNeed(container: HTMLElement): HTMLElement | null;
 
    public wrapTheadAndTbodyIntoTableIfNeed(container: HTMLElement): HTMLElement | null;
 
    public wrapTrsIntoTbodyIfNeed(container: HTMLElement): HTMLElement | null;
  }
 
  class WwTableSelectionManager {
    constructor(wwe: WysiwygEditor);
 
    public createRangeBySelectedCells(): void;
 
    public destroy(): void;
 
    public getSelectedCells(): HTMLElement;
 
    public getSelectionRangeFromTable(
      selectionStart: HTMLElement,
      selectionEnd: HTMLElement
    ): SelectionRange;
 
    public highlightTableCellsBy(selectionStart: HTMLElement, selectionEnd: HTMLElement): void;
 
    public removeClassAttrbuteFromAllCellsIfNeed(): void;
 
    public setTableSelectionTimerIfNeed(selectionStart: HTMLElement): void;
 
    public styleToSelectedCells(onStyle: SquireExt, options?: object): void;
  }
 
  (/, @TODO:, change, toastMark, type, definition, to, @toast-ui/toastmark, type, file, through, importing)
  class MarkdownEditor {
    static factory(
      el: HTMLElement,
      eventManager: EventManager,
      toastMark: any,
      options: MarkdownEditorOptions
    ): MarkdownEditor;
 
    constructor(
      el: HTMLElement,
      eventManager: EventManager,
      toastMark: any,
      options: MarkdownEditorOptions
    );
 
    public getTextObject(range: Range | RangeType): MdTextObject;
 
    public setValue(markdown: string, cursorToEnd?: boolean): void;
 
    public resetState(): void;
 
    public getMdDocument(): any;
  }
 
  class WysiwygEditor {
    static factory(
      el: HTMLElement,
      eventManager: EventManager,
      options: WysiwygEditorOptions
    ): WysiwygEditor;
 
    constructor(el: HTMLElement, eventManager: EventManager, options: WysiwygEditorOptions);
 
    public addKeyEventHandler(keyMap: string | string[], handler: HandlerFunc): void;
 
    public addWidget(range: Range, node: Node, style: string, offset?: number): void;
 
    public blur(): void;
 
    public breakToNewDefaultBlock(range: Range, where?: string): void;
 
    public changeBlockFormatTo(targetTagName: string): void;
 
    public findTextNodeFilter(): boolean;
 
    public fixIMERange(): void;
 
    public focus(): void;
 
    public getEditor(): SquireExt;
 
    public getIMERange(): Range;
 
    public getRange(): Range;
 
    public getTextObject(range: Range): WwTextObject;
 
    public getValue(): string;
 
    public hasFormatWithRx(rx: RegExp): boolean;
 
    public init(useDefaultHTMLSanitizer: boolean): void;
 
    public insertText(text: string): void;
 
    public makeEmptyBlockCurrentSelection(): void;
 
    public moveCursorToEnd(): void;
 
    public moveCursorToStart(): void;
 
    public postProcessForChange(): void;
 
    public readySilentChange(): void;
 
    public remove(): void;
 
    public removeKeyEventHandler(keyMap: string, handler: HandlerFunc): void;
 
    public replaceContentText(container: Node, from: string, to: string): void;
 
    public replaceRelativeOffset(content: string, offset: number, overwriteLength: number): void;
 
    public replaceSelection(content: string, range: Range): void;
 
    public reset(): void;
 
    public restoreSavedSelection(): void;
 
    public saveSelection(range: Range): void;
 
    public scrollTop(value: number): boolean;
 
    public setHeight(height: number | string): void;
 
    public setPlaceholder(placeholder: string): void;
 
    public setMinHeight(minHeight: number): void;
 
    public setRange(range: Range): void;
 
    public getLinkAttribute(): LinkAttribute;
 
    public setSelectionByContainerAndOffset(
      startContainer: Node,
      startOffset: number,
      endContainer: Node,
      endOffset: number
    ): Range;
 
    public setValue(html: string, cursorToEnd?: boolean): void;
 
    public unwrapBlockTag(condition?: (tagName: string) => boolean): void;
 
    public getBody(): HTMLElement;
 
    public scrollIntoCursor(): void;
 
    public isInTable(range: Range): boolean;
  }
 
  class EventManager {
    public addEventType(type: string): void;
 
    public emit(eventName: string): any[];
 
    public emitReduce(eventName: string, sourceText: string): string;
 
    public listen(typeStr: string, handler: HandlerFunc): void;
 
    public removeEventHandler(typeStr: string, handler?: HandlerFunc): void;
  }
 
  export class Editor {
    public static codeBlockManager: CodeBlockManager;
 
    public static CommandManager: CommandManager;
 
    public static isViewer: boolean;
 
    public static WwCodeBlockManager: WwCodeBlockManager;
 
    public static WwTableManager: WwTableManager;
 
    public static WwTableSelectionManager: WwTableSelectionManager;
 
    public static factory(options: EditorOptions): Editor | Viewer;
 
    public static getInstances(): Editor[];
 
    public static setLanguage(code: string, data: LanguageData): void;
 
    constructor(options: EditorOptions);
 
    public addHook(type: string, handler: HandlerFunc): void;
 
    public addWidget(selection: Range, node: Node, style: string, offset?: number): void;
 
    public afterAddedCommand(): void;
 
    public blur(): void;
 
    public changeMode(mode: string, isWithoutFocus?: boolean): void;
 
    public changePreviewStyle(style: PreviewStyle): void;
 
    public exec(name: string, ...args: any[]): void;
 
    public focus(): void;
 
    public getCodeMirror(): CodeMirrorType;
 
    public getCurrentModeEditor(): MarkdownEditor | WysiwygEditor;
 
    public getCurrentPreviewStyle(): PreviewStyle;
 
    public getHtml(): string;
 
    public getMarkdown(): string;
 
    public getRange(): Range | RangeType;
 
    public getSelectedText(): string;
 
    public getSquire(): SquireExt;
 
    public getTextObject(range: Range | RangeType): MdTextObject | WwTextObject;
 
    public getUI(): UI;
 
    public getValue(): string;
 
    public height(height: string): string;
 
    public hide(): void;
 
    public insertText(text: string): void;
 
    public isMarkdownMode(): boolean;
 
    public isViewer(): boolean;
 
    public isWysiwygMode(): boolean;
 
    public minHeight(minHeight: string): string;
 
    public moveCursorToEnd(): void;
 
    public moveCursorToStart(): void;
 
    public off(type: string): void;
 
    public on(type: string, handler: HandlerFunc): void;
 
    public remove(): void;
 
    public removeHook(type: string): void;
 
    public reset(): void;
 
    public scrollTop(value: number): number;
 
    public setHtml(html: string, cursorToEnd?: boolean): void;
 
    public setMarkdown(markdown: string, cursorToEnd?: boolean): void;
 
    public setUI(UI: UI): void;
 
    public setValue(value: string, cursorToEnd?: boolean): void;
 
    public show(): void;
 
    public setCodeBlockLanguages(languages?: string[]): void;
  }
 
  export class Viewer {
    public static isViewer: boolean;
 
    public static codeBlockManager: CodeBlockManager;
 
    public static WwCodeBlockManager: null;
 
    public static WwTableManager: null;
 
    public static WwTableSelectionManager: null;
 
    constructor(options: ViewerOptions);
 
    public addHook(type: string, handler: HandlerFunc): void;
 
    public isMarkdownMode(): boolean;
 
    public isViewer(): boolean;
 
    public isWysiwygMode(): boolean;
 
    public off(type: string): void;
 
    public on(type: string, handler: HandlerFunc): void;
 
    public remove(): void;
 
    public setMarkdown(markdown: string): void;
 
    public setValue(markdown: string): void;
 
    public setCodeBlockLanguages(languages?: string[]): void;
  }
}
 
declare module '@toast-ui/editor' {
  export type EditorOptions = toastui.EditorOptions;
  export type CustomConvertor = toastui.ConvertorClass;
  export type EventMap = toastui.EventMap;
  export type EditorHookMap = toastui.EditorHookMap;
  export type CustomHTMLRenderer = toastui.CustomHTMLRenderer;
  export type ExtendedAutolinks = toastui.ExtendedAutolinks;
  export type LinkAttribute = toastui.LinkAttribute;
  export default toastui.Editor;
}
 
declare module '@toast-ui/editor/dist/toastui-editor-viewer' {
  export type ViewerOptions = toastui.ViewerOptions;
  export type CustomConvertor = toastui.ConvertorClass;
  export type EventMap = toastui.EventMap;
  export type ViewerHookMap = toastui.ViewerHookMap;
  export type CustomHTMLRenderer = toastui.CustomHTMLRenderer;
  export type ExtendedAutolinks = toastui.ExtendedAutolinks;
  export type LinkAttribute = toastui.LinkAttribute;
  export default toastui.Viewer;
}
 


# In[ ]:


{
  "name": "@toast-ui/editor",
  "version": "2.1.0",
  "description": "GFM  Markdown Wysiwyg Editor - Productive and Extensible",
  "keywords": [
    "nhn",
    "toast",
    "toastui",
    "toast-ui",
    "markdown",
    "wysiwyg",
    "editor",
    "preview",
    "gfm"
  ],
  "main": "dist/toastui-editor.js",
  "files": [
    "dist/*.js",
    "dist/*.css",
    "dist/i18n",
    "index.d.ts"
  ],
  "author": "NHN FE Development Lab <dl_javascript@nhn.com>",
  "license": "MIT",
  "repository": {
    "type": "git",
    "url": "https://github.com/nhn/tui.editor.git",
    "directory": "apps/editor"
  },
  "bugs": {
    "url": "https://github.com/nhn/tui.editor/issues"
  },
  "homepage": "https://ui.toast.com",
  "browserslist": "last 2 versions, not ie <= 10",
  "scripts": {
    "lint": "eslint .",
    "test": "karma start --no-single-run",
    "test:ne": "cross-env KARMA_SERVER=ne karma start",
    "test:types": "tsc --project test/types",
    "e2e": "testcafe chrome 'test/e2e/**/*.spec.js'",
    "e2e:sl": "testcafe \"saucelabs:Chrome@65.0:Windows 10,saucelabs:Firefox@59.0:Windows 10,saucelabs:Safari@10.0:OS X 10.11,saucelabs:Internet Explorer@11.103:Windows 10,saucelabs:MicrosoftEdge@16.16299:Windows 10\" 'test/e2e/**/*.spec.js'",
    "serve": "webpack-dev-server",
    "serve:viewer": "webpack-dev-server --viewer",
    "serve:all": "webpack-dev-server --all",
    "build:i18n": "cross-env webpack --config scripts/webpack.config.i18n.js && webpack --config scripts/webpack.config.i18n.js --minify",
    "build:prod": "cross-env webpack --mode=production && webpack --mode=production --minify && node tsBannerGenerator.js",
    "build": "npm run build:i18n && npm run build:prod",
    "note": "tui-note --tag=$(git describe --tags)",
    "tslint": "tslint index.d.ts",
    "doc:serve": "tuidoc --serv",
    "doc": "tuidoc"
  },
  "devDependencies": {
    "@babel/core": "^7.8.3",
    "@babel/plugin-proposal-class-properties": "^7.8.3",
    "@babel/preset-env": "^7.8.3",
    "@toast-ui/release-notes": "^2.0.1",
    "@toast-ui/squire": "file:../../libs/squire",
    "@toast-ui/to-mark": "file:../../libs/to-mark",
    "@toast-ui/toastmark": "file:../../libs/toastmark",
    "babel-eslint": "^10.0.3",
    "babel-loader": "^8.0.6",
    "babel-plugin-istanbul": "^6.0.0",
    "cross-env": "^6.0.3",
    "css-loader": "^3.4.2",
    "eslint": "^6.8.0",
    "eslint-config-prettier": "^6.9.0",
    "eslint-config-tui": "^3.0.0",
    "eslint-loader": "^3.0.3",
    "eslint-plugin-prettier": "^3.1.2",
    "filemanager-webpack-plugin": "^2.0.5",
    "istanbul-instrumenter-loader": "^3.0.1",
    "jasmine-core": "^2.99.1",
    "jquery": "^3.4.1",
    "karma": "^4.4.1",
    "karma-chrome-launcher": "^3.1.0",
    "karma-coverage-istanbul-reporter": "^2.1.1",
    "karma-jasmine": "^1.1.2",
    "karma-jasmine-ajax": "^0.1.13",
    "karma-jasmine-jquery": "^0.1.1",
    "karma-sourcemap-loader": "^0.3.7",
    "karma-webdriver-launcher": "github:nhn/karma-webdriver-launcher#v1.2.0",
    "karma-webpack": "^4.0.2",
    "mini-css-extract-plugin": "^0.9.0",
    "optimize-css-assets-webpack-plugin": "^5.0.3",
    "prettier": "^1.19.1",
    "resize-observer-polyfill": "^1.5.1",
    "terser-webpack-plugin": "^2.2.1",
    "testcafe": "^0.23.3",
    "testcafe-browser-provider-saucelabs": "^1.3.0",
    "tslint": "^5.12.0",
    "tui-code-snippet": "^2.3.1",
    "typescript": "^3.2.2",
    "url-loader": "^3.0.0",
    "webpack": "^4.40.2",
    "webpack-bundle-analyzer": "^3.6.0",
    "webpack-cli": "^3.3.9",
    "webpack-dev-server": "^3.1.11",
    "webpack-glob-entry": "^2.1.1"
  },
  "dependencies": {
    "@types/codemirror": "0.0.71",
    "codemirror": "^5.48.4"
  }
}
 


# In[ ]:


**()
 * @fileoverview Configs for i18n bundle file
 * @author NHN FE Development Lab <dl_javascript@nhn.com>
 */
const path = require('path');
const webpack = require('webpack');
const entry = require('webpack-glob-entry');
const pkg = require('../package.json');
 
const TerserPlugin = require('terser-webpack-plugin');
const FileManagerPlugin = require('filemanager-webpack-plugin');
 
function getOptimizationConfig(minify) {
  const minimizer = [];
 
  if (minify) {
    minimizer.push(
      new TerserPlugin({
        cache: true,
        parallel: true,
        sourceMap: false,
        extractComments: false
      })
    );
  }
 
  return { minimizer };
}
 
function getEntries() {
  const entries = entry('./src/js/i18n/*.js');
 
  delete entries['en-us'];
 
  return entries;
}
 
module.exports = (env, argv) => {
  const minify = !!argv.minify;
 
  return {
    mode: 'production',
    entry: getEntries(),
    output: {
      libraryTarget: 'umd',
      path: path.resolve(__dirname, minify ? '../dist/cdn/i18n' : '../dist/i18n'),
      filename: `[name]${minify ? '.min' : ''}.js`
    },
    externals: [
      {
        '../editor': {
          commonjs: '@toast-ui/editor',
          commonjs2: '@toast-ui/editor',
          amd: '@toast-ui/editor',
          root: ['toastui', 'Editor']
        }
      }
    ],
    module: {
      rules: [
        {
          test: /\.js$/,
          exclude: /node_modules/,
          loader: 'eslint-loader',
          enforce: 'pre',
          options: {
            failOnError: true
          }
        },
        {
          test: /\.js$/,
          exclude: /node_modules|dist/,
          loader: 'babel-loader?cacheDirectory',
          options: {
            rootMode: 'upward'
          }
        }
      ]
    },
    plugins: [
      new webpack.BannerPlugin(
        [
          'TOAST UI Editor : i18n',
          `@version ${pkg.version}`,
          `@author ${pkg.author}`,
          `@license ${pkg.license}`
        ].join('\n')
      ),
      new FileManagerPlugin({
        onEnd: {
          copy: [{ source: './dist/i18n/*.js', destination: './dist/cdn/i18n' }]
        }
      })
    ],
    optimization: getOptimizationConfig(minify)
  };
};
 

