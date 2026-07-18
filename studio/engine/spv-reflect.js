/*
 * slangfx-web — minimal SPIR-V reflector.
 *
 * Port of src/spv_reflect.c, adapted for the WebGPU rewrites: after
 * preprocessing there are no push constants; the former Push block is a
 * std140 UBO at set=1 binding=0 and the classic slang UBO sits at set=0
 * binding=0. We recover both blocks' member names, byte offsets, and a
 * type-accurate total size (needed for GPUBuffer minBindingSize).
 */

const Op = {
  Name: 5,
  MemberName: 6,
  TypeInt: 21,
  TypeFloat: 22,
  TypeVector: 23,
  TypeMatrix: 24,
  TypeArray: 28,
  TypeStruct: 30,
  TypePointer: 32,
  Constant: 43,
  Variable: 59,
  Decorate: 71,
  MemberDecorate: 72,
};

const Decoration = { Block: 2, ArrayStride: 6, Binding: 33, DescriptorSet: 34, Offset: 35, MatrixStride: 7 };
const StorageClass = { UniformConstant: 0, Uniform: 2, PushConstant: 9 };

function readString(words, start, end) {
  const bytes = [];
  for (let i = start; i < end; i++) {
    const w = words[i];
    for (let b = 0; b < 4; b++) {
      const c = (w >>> (b * 8)) & 0xff;
      if (c === 0) return String.fromCharCode(...bytes);
      bytes.push(c);
    }
  }
  return String.fromCharCode(...bytes);
}

/**
 * Reflect uniform block layouts from a SPIR-V binary.
 * @param {Uint32Array} spv
 * @returns {{ubo: Block|null, push: Block|null}} where Block =
 *          {size, members: [{name, offset, size}]}
 *          `ubo` = block at set 0 binding 0, `push` = block at set 1 binding 0.
 */
export function reflectSpirv(spv) {
  if (spv.length < 5 || spv[0] !== 0x07230203) throw new Error('bad SPIR-V');
  const bound = spv[3];

  const names = new Map();          // id -> string
  const memberNames = new Map();    // structId -> Map(memberIdx -> string)
  const memberOffsets = new Map();  // structId -> Map(memberIdx -> offset)
  const structMembers = new Map();  // structId -> [memberTypeId]
  const pointers = new Map();       // ptrId -> {storage, pointee}
  const variables = [];             // {id, ptrType, storage}
  const decor = new Map();          // id -> {set, binding, block}
  const typeInfo = new Map();       // typeId -> {kind, ...}
  const constants = new Map();      // constId -> value (int)
  const arrayStrides = new Map();   // arrayTypeId -> stride

  let i = 5;
  while (i < spv.length) {
    const opWord = spv[i];
    const opcode = opWord & 0xffff;
    const wcount = opWord >>> 16;
    if (wcount === 0 || i + wcount > spv.length) throw new Error(`SPIR-V truncated at word ${i}`);

    switch (opcode) {
      case Op.Name:
        if (wcount >= 3) names.set(spv[i + 1], readString(spv, i + 2, i + wcount));
        break;
      case Op.MemberName:
        if (wcount >= 4) {
          const sid = spv[i + 1];
          if (!memberNames.has(sid)) memberNames.set(sid, new Map());
          memberNames.get(sid).set(spv[i + 2], readString(spv, i + 3, i + wcount));
        }
        break;
      case Op.Decorate:
        if (wcount >= 3) {
          const target = spv[i + 1];
          const d = decor.get(target) ?? {};
          const kind = spv[i + 2];
          if (kind === Decoration.Block) d.block = true;
          else if (kind === Decoration.DescriptorSet && wcount >= 4) d.set = spv[i + 3];
          else if (kind === Decoration.Binding && wcount >= 4) d.binding = spv[i + 3];
          else if (kind === Decoration.ArrayStride && wcount >= 4) arrayStrides.set(target, spv[i + 3]);
          decor.set(target, d);
        }
        break;
      case Op.MemberDecorate:
        if (wcount >= 5 && spv[i + 3] === Decoration.Offset) {
          const sid = spv[i + 1];
          if (!memberOffsets.has(sid)) memberOffsets.set(sid, new Map());
          memberOffsets.get(sid).set(spv[i + 2], spv[i + 4]);
        }
        break;
      case Op.TypeInt:
      case Op.TypeFloat:
        typeInfo.set(spv[i + 1], { kind: 'scalar', size: 4 });
        break;
      case Op.TypeVector:
        typeInfo.set(spv[i + 1], { kind: 'vector', component: spv[i + 2], count: spv[i + 3] });
        break;
      case Op.TypeMatrix:
        typeInfo.set(spv[i + 1], { kind: 'matrix', column: spv[i + 2], count: spv[i + 3] });
        break;
      case Op.TypeArray:
        typeInfo.set(spv[i + 1], { kind: 'array', element: spv[i + 2], lengthId: spv[i + 3] });
        break;
      case Op.TypeStruct: {
        const sid = spv[i + 1];
        const members = [];
        for (let k = 2; k < wcount; k++) members.push(spv[i + k]);
        structMembers.set(sid, members);
        break;
      }
      case Op.TypePointer:
        if (wcount >= 4) pointers.set(spv[i + 1], { storage: spv[i + 2], pointee: spv[i + 3] });
        break;
      case Op.Constant:
        // op[1]=result type, op[2]=result id, op[3]=value (32-bit)
        if (wcount >= 4) constants.set(spv[i + 2], spv[i + 3]);
        break;
      case Op.Variable:
        if (wcount >= 4) variables.push({ ptrType: spv[i + 1], id: spv[i + 2], storage: spv[i + 3] });
        break;
      default:
        break;
    }
    i += wcount;
  }

  function typeSize(tid) {
    const t = typeInfo.get(tid);
    if (!t) return 4;
    switch (t.kind) {
      case 'scalar': return 4;
      case 'vector': return 4 * t.count;
      case 'matrix': {
        // std140 column stride is 16 for vec3/vec4 columns, matching slang use.
        const colInfo = typeInfo.get(t.column);
        const stride = colInfo && colInfo.count <= 2 ? 8 : 16;
        return stride * t.count;
      }
      case 'array': {
        const len = constants.get(t.lengthId) ?? 1;
        const stride = arrayStrides.get(tid) ?? typeSize(t.element);
        return len * stride;
      }
      default: return 4;
    }
  }

  function emitBlock(structId) {
    const memberTypes = structMembers.get(structId) ?? [];
    const offsets = memberOffsets.get(structId) ?? new Map();
    const mnames = memberNames.get(structId) ?? new Map();
    const members = [];
    let total = 0;
    for (let k = 0; k < memberTypes.length; k++) {
      const offset = offsets.get(k) ?? 0;
      const size = typeSize(memberTypes[k]);
      members.push({ name: mnames.get(k) ?? null, offset, size });
      total = Math.max(total, offset + size);
    }
    return { size: (total + 15) & ~15, members };
  }

  let ubo = null;
  let push = null;
  for (const v of variables) {
    const ptr = pointers.get(v.ptrType);
    if (!ptr) continue;
    const d = decor.get(v.id) ?? {};
    const structIsBlock = (decor.get(ptr.pointee) ?? {}).block;
    if (v.storage === StorageClass.Uniform && structIsBlock) {
      if (d.set === 0 && d.binding === 0 && !ubo) ubo = emitBlock(ptr.pointee);
      else if (d.set === 1 && d.binding === 0 && !push) push = emitBlock(ptr.pointee);
    } else if (v.storage === StorageClass.PushConstant && !push) {
      // Shouldn't occur after the preprocess rewrite, but reflect anyway.
      push = emitBlock(ptr.pointee);
    }
  }
  return { ubo, push };
}
