import { WireType } from "@protobuf-ts/runtime";
import { UnknownFieldHandler } from "@protobuf-ts/runtime";
import { reflectionMergePartial } from "@protobuf-ts/runtime";
import { MessageType } from "@protobuf-ts/runtime";
// @generated message type with reflection information, may provide speed optimized methods
class StreamPosition$Type extends MessageType {
    constructor() {
        super("s2.v1.StreamPosition", [
            { no: 1, name: "seq_num", kind: "scalar", T: 4 /*ScalarType.UINT64*/, L: 0 /*LongType.BIGINT*/ },
            { no: 2, name: "timestamp", kind: "scalar", T: 4 /*ScalarType.UINT64*/, L: 0 /*LongType.BIGINT*/ }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.seqNum = 0n;
        message.timestamp = 0n;
        if (value !== undefined)
            reflectionMergePartial(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* uint64 seq_num */ 1:
                    message.seqNum = reader.uint64().toBigInt();
                    break;
                case /* uint64 timestamp */ 2:
                    message.timestamp = reader.uint64().toBigInt();
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
            }
        }
        return message;
    }
    internalBinaryWrite(message, writer, options) {
        /* uint64 seq_num = 1; */
        if (message.seqNum !== 0n)
            writer.tag(1, WireType.Varint).uint64(message.seqNum);
        /* uint64 timestamp = 2; */
        if (message.timestamp !== 0n)
            writer.tag(2, WireType.Varint).uint64(message.timestamp);
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message s2.v1.StreamPosition
 */
export const StreamPosition = new StreamPosition$Type();
// @generated message type with reflection information, may provide speed optimized methods
class Header$Type extends MessageType {
    constructor() {
        super("s2.v1.Header", [
            { no: 1, name: "name", kind: "scalar", T: 12 /*ScalarType.BYTES*/ },
            { no: 2, name: "value", kind: "scalar", T: 12 /*ScalarType.BYTES*/ }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.name = new Uint8Array(0);
        message.value = new Uint8Array(0);
        if (value !== undefined)
            reflectionMergePartial(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* bytes name */ 1:
                    message.name = reader.bytes();
                    break;
                case /* bytes value */ 2:
                    message.value = reader.bytes();
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
            }
        }
        return message;
    }
    internalBinaryWrite(message, writer, options) {
        /* bytes name = 1; */
        if (message.name.length)
            writer.tag(1, WireType.LengthDelimited).bytes(message.name);
        /* bytes value = 2; */
        if (message.value.length)
            writer.tag(2, WireType.LengthDelimited).bytes(message.value);
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message s2.v1.Header
 */
export const Header = new Header$Type();
// @generated message type with reflection information, may provide speed optimized methods
class AppendRecord$Type extends MessageType {
    constructor() {
        super("s2.v1.AppendRecord", [
            { no: 1, name: "timestamp", kind: "scalar", opt: true, T: 4 /*ScalarType.UINT64*/, L: 0 /*LongType.BIGINT*/ },
            { no: 2, name: "headers", kind: "message", repeat: 2 /*RepeatType.UNPACKED*/, T: () => Header },
            { no: 3, name: "body", kind: "scalar", T: 12 /*ScalarType.BYTES*/ }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.headers = [];
        message.body = new Uint8Array(0);
        if (value !== undefined)
            reflectionMergePartial(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* optional uint64 timestamp */ 1:
                    message.timestamp = reader.uint64().toBigInt();
                    break;
                case /* repeated s2.v1.Header headers */ 2:
                    message.headers.push(Header.internalBinaryRead(reader, reader.uint32(), options));
                    break;
                case /* bytes body */ 3:
                    message.body = reader.bytes();
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
            }
        }
        return message;
    }
    internalBinaryWrite(message, writer, options) {
        /* optional uint64 timestamp = 1; */
        if (message.timestamp !== undefined)
            writer.tag(1, WireType.Varint).uint64(message.timestamp);
        /* repeated s2.v1.Header headers = 2; */
        for (let i = 0; i < message.headers.length; i++)
            Header.internalBinaryWrite(message.headers[i], writer.tag(2, WireType.LengthDelimited).fork(), options).join();
        /* bytes body = 3; */
        if (message.body.length)
            writer.tag(3, WireType.LengthDelimited).bytes(message.body);
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message s2.v1.AppendRecord
 */
export const AppendRecord = new AppendRecord$Type();
// @generated message type with reflection information, may provide speed optimized methods
class AppendInput$Type extends MessageType {
    constructor() {
        super("s2.v1.AppendInput", [
            { no: 1, name: "records", kind: "message", repeat: 2 /*RepeatType.UNPACKED*/, T: () => AppendRecord },
            { no: 2, name: "match_seq_num", kind: "scalar", opt: true, T: 4 /*ScalarType.UINT64*/, L: 0 /*LongType.BIGINT*/ },
            { no: 3, name: "fencing_token", kind: "scalar", opt: true, T: 9 /*ScalarType.STRING*/ }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.records = [];
        if (value !== undefined)
            reflectionMergePartial(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* repeated s2.v1.AppendRecord records */ 1:
                    message.records.push(AppendRecord.internalBinaryRead(reader, reader.uint32(), options));
                    break;
                case /* optional uint64 match_seq_num */ 2:
                    message.matchSeqNum = reader.uint64().toBigInt();
                    break;
                case /* optional string fencing_token */ 3:
                    message.fencingToken = reader.string();
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
            }
        }
        return message;
    }
    internalBinaryWrite(message, writer, options) {
        /* repeated s2.v1.AppendRecord records = 1; */
        for (let i = 0; i < message.records.length; i++)
            AppendRecord.internalBinaryWrite(message.records[i], writer.tag(1, WireType.LengthDelimited).fork(), options).join();
        /* optional uint64 match_seq_num = 2; */
        if (message.matchSeqNum !== undefined)
            writer.tag(2, WireType.Varint).uint64(message.matchSeqNum);
        /* optional string fencing_token = 3; */
        if (message.fencingToken !== undefined)
            writer.tag(3, WireType.LengthDelimited).string(message.fencingToken);
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message s2.v1.AppendInput
 */
export const AppendInput = new AppendInput$Type();
// @generated message type with reflection information, may provide speed optimized methods
class AppendAck$Type extends MessageType {
    constructor() {
        super("s2.v1.AppendAck", [
            { no: 1, name: "start", kind: "message", T: () => StreamPosition },
            { no: 2, name: "end", kind: "message", T: () => StreamPosition },
            { no: 3, name: "tail", kind: "message", T: () => StreamPosition }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        if (value !== undefined)
            reflectionMergePartial(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* s2.v1.StreamPosition start */ 1:
                    message.start = StreamPosition.internalBinaryRead(reader, reader.uint32(), options, message.start);
                    break;
                case /* s2.v1.StreamPosition end */ 2:
                    message.end = StreamPosition.internalBinaryRead(reader, reader.uint32(), options, message.end);
                    break;
                case /* s2.v1.StreamPosition tail */ 3:
                    message.tail = StreamPosition.internalBinaryRead(reader, reader.uint32(), options, message.tail);
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
            }
        }
        return message;
    }
    internalBinaryWrite(message, writer, options) {
        /* s2.v1.StreamPosition start = 1; */
        if (message.start)
            StreamPosition.internalBinaryWrite(message.start, writer.tag(1, WireType.LengthDelimited).fork(), options).join();
        /* s2.v1.StreamPosition end = 2; */
        if (message.end)
            StreamPosition.internalBinaryWrite(message.end, writer.tag(2, WireType.LengthDelimited).fork(), options).join();
        /* s2.v1.StreamPosition tail = 3; */
        if (message.tail)
            StreamPosition.internalBinaryWrite(message.tail, writer.tag(3, WireType.LengthDelimited).fork(), options).join();
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message s2.v1.AppendAck
 */
export const AppendAck = new AppendAck$Type();
// @generated message type with reflection information, may provide speed optimized methods
class SequencedRecord$Type extends MessageType {
    constructor() {
        super("s2.v1.SequencedRecord", [
            { no: 1, name: "seq_num", kind: "scalar", T: 4 /*ScalarType.UINT64*/, L: 0 /*LongType.BIGINT*/ },
            { no: 2, name: "timestamp", kind: "scalar", T: 4 /*ScalarType.UINT64*/, L: 0 /*LongType.BIGINT*/ },
            { no: 3, name: "headers", kind: "message", repeat: 2 /*RepeatType.UNPACKED*/, T: () => Header },
            { no: 4, name: "body", kind: "scalar", T: 12 /*ScalarType.BYTES*/ }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.seqNum = 0n;
        message.timestamp = 0n;
        message.headers = [];
        message.body = new Uint8Array(0);
        if (value !== undefined)
            reflectionMergePartial(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* uint64 seq_num */ 1:
                    message.seqNum = reader.uint64().toBigInt();
                    break;
                case /* uint64 timestamp */ 2:
                    message.timestamp = reader.uint64().toBigInt();
                    break;
                case /* repeated s2.v1.Header headers */ 3:
                    message.headers.push(Header.internalBinaryRead(reader, reader.uint32(), options));
                    break;
                case /* bytes body */ 4:
                    message.body = reader.bytes();
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
            }
        }
        return message;
    }
    internalBinaryWrite(message, writer, options) {
        /* uint64 seq_num = 1; */
        if (message.seqNum !== 0n)
            writer.tag(1, WireType.Varint).uint64(message.seqNum);
        /* uint64 timestamp = 2; */
        if (message.timestamp !== 0n)
            writer.tag(2, WireType.Varint).uint64(message.timestamp);
        /* repeated s2.v1.Header headers = 3; */
        for (let i = 0; i < message.headers.length; i++)
            Header.internalBinaryWrite(message.headers[i], writer.tag(3, WireType.LengthDelimited).fork(), options).join();
        /* bytes body = 4; */
        if (message.body.length)
            writer.tag(4, WireType.LengthDelimited).bytes(message.body);
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message s2.v1.SequencedRecord
 */
export const SequencedRecord = new SequencedRecord$Type();
// @generated message type with reflection information, may provide speed optimized methods
class ReadBatch$Type extends MessageType {
    constructor() {
        super("s2.v1.ReadBatch", [
            { no: 1, name: "records", kind: "message", repeat: 2 /*RepeatType.UNPACKED*/, T: () => SequencedRecord },
            { no: 2, name: "tail", kind: "message", T: () => StreamPosition }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.records = [];
        if (value !== undefined)
            reflectionMergePartial(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* repeated s2.v1.SequencedRecord records */ 1:
                    message.records.push(SequencedRecord.internalBinaryRead(reader, reader.uint32(), options));
                    break;
                case /* optional s2.v1.StreamPosition tail */ 2:
                    message.tail = StreamPosition.internalBinaryRead(reader, reader.uint32(), options, message.tail);
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
            }
        }
        return message;
    }
    internalBinaryWrite(message, writer, options) {
        /* repeated s2.v1.SequencedRecord records = 1; */
        for (let i = 0; i < message.records.length; i++)
            SequencedRecord.internalBinaryWrite(message.records[i], writer.tag(1, WireType.LengthDelimited).fork(), options).join();
        /* optional s2.v1.StreamPosition tail = 2; */
        if (message.tail)
            StreamPosition.internalBinaryWrite(message.tail, writer.tag(2, WireType.LengthDelimited).fork(), options).join();
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message s2.v1.ReadBatch
 */
export const ReadBatch = new ReadBatch$Type();
//# sourceMappingURL=s2.js.map