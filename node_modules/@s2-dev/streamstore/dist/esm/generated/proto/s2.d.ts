import type { BinaryWriteOptions } from "@protobuf-ts/runtime";
import type { IBinaryWriter } from "@protobuf-ts/runtime";
import type { BinaryReadOptions } from "@protobuf-ts/runtime";
import type { IBinaryReader } from "@protobuf-ts/runtime";
import type { PartialMessage } from "@protobuf-ts/runtime";
import { MessageType } from "@protobuf-ts/runtime";
/**
 * Position of a record in a stream.
 *
 * @generated from protobuf message s2.v1.StreamPosition
 */
export interface StreamPosition {
    /**
     * Sequence number assigned by the service.
     *
     * @generated from protobuf field: uint64 seq_num = 1
     */
    seqNum: bigint;
    /**
     * Timestamp, which may be user-specified or assigned by the service.
     * If it is assigned by the service, it will represent milliseconds since Unix epoch.
     *
     * @generated from protobuf field: uint64 timestamp = 2
     */
    timestamp: bigint;
}
/**
 * Headers add structured information to a record as name-value pairs.
 *
 * @generated from protobuf message s2.v1.Header
 */
export interface Header {
    /**
     * Header name blob.
     * The name cannot be empty, with the exception of an S2 command record.
     *
     * @generated from protobuf field: bytes name = 1
     */
    name: Uint8Array;
    /**
     * Header value blob.
     *
     * @generated from protobuf field: bytes value = 2
     */
    value: Uint8Array;
}
/**
 * Record to be appended to a stream.
 *
 * @generated from protobuf message s2.v1.AppendRecord
 */
export interface AppendRecord {
    /**
     * Timestamp for this record.
     * Precise semantics depend on the stream's `timestamping` config.
     *
     * @generated from protobuf field: optional uint64 timestamp = 1
     */
    timestamp?: bigint;
    /**
     * Series of name-value pairs for this record.
     *
     * @generated from protobuf field: repeated s2.v1.Header headers = 2
     */
    headers: Header[];
    /**
     * Body of this record.
     *
     * @generated from protobuf field: bytes body = 3
     */
    body: Uint8Array;
}
/**
 * Payload of an Append request message.
 *
 * @generated from protobuf message s2.v1.AppendInput
 */
export interface AppendInput {
    /**
     * Batch of records to append atomically, which must contain at least one record, and no more than 1000.
     * The total size of a batch of records may not exceed 1MiB of metered bytes.
     *
     * @generated from protobuf field: repeated s2.v1.AppendRecord records = 1
     */
    records: AppendRecord[];
    /**
     * Enforce that the sequence number issued to the first record matches.
     *
     * @generated from protobuf field: optional uint64 match_seq_num = 2
     */
    matchSeqNum?: bigint;
    /**
     * Enforce a fencing token which must have been previously set by a `fence` command record.
     *
     * @generated from protobuf field: optional string fencing_token = 3
     */
    fencingToken?: string;
}
/**
 * Success response message to an Append request.
 *
 * @generated from protobuf message s2.v1.AppendAck
 */
export interface AppendAck {
    /**
     * Sequence number and timestamp of the first record that was appended.
     *
     * @generated from protobuf field: s2.v1.StreamPosition start = 1
     */
    start?: StreamPosition;
    /**
     * Sequence number of the last record that was appended + 1, and timestamp of the last record that was appended.
     * The difference between `end.seq_num` and `start.seq_num` will be the number of records appended.
     *
     * @generated from protobuf field: s2.v1.StreamPosition end = 2
     */
    end?: StreamPosition;
    /**
     * Sequence number that will be assigned to the next record on the stream, and timestamp of the last record on the stream.
     * This can be greater than the `end` position in case of concurrent appends.
     *
     * @generated from protobuf field: s2.v1.StreamPosition tail = 3
     */
    tail?: StreamPosition;
}
/**
 * Record that is durably sequenced on a stream.
 *
 * @generated from protobuf message s2.v1.SequencedRecord
 */
export interface SequencedRecord {
    /**
     * Sequence number assigned to this record.
     *
     * @generated from protobuf field: uint64 seq_num = 1
     */
    seqNum: bigint;
    /**
     * Timestamp for this record.
     *
     * @generated from protobuf field: uint64 timestamp = 2
     */
    timestamp: bigint;
    /**
     * Series of name-value pairs for this record.
     *
     * @generated from protobuf field: repeated s2.v1.Header headers = 3
     */
    headers: Header[];
    /**
     * Body of this record.
     *
     * @generated from protobuf field: bytes body = 4
     */
    body: Uint8Array;
}
/**
 * Success response message to a Read request.
 *
 * @generated from protobuf message s2.v1.ReadBatch
 */
export interface ReadBatch {
    /**
     * Records that are durably sequenced on the stream, retrieved based on the requested criteria.
     * This can only be empty in response to a unary read if the request cannot be satisfied without violating an explicit bound (`count`, `bytes`, or `until`).
     * In the context of a session, it can be empty as a heartbeat message. A heartbeat will be sent whenever a switch to following in real-time happens, and then at a randomized gap between 5 and 15 seconds if no records have become available.
     *
     * @generated from protobuf field: repeated s2.v1.SequencedRecord records = 1
     */
    records: SequencedRecord[];
    /**
     * Sequence number that will be assigned to the next record on the stream, and timestamp of the last record.
     * It will only be present when reading recent records.
     *
     * @generated from protobuf field: optional s2.v1.StreamPosition tail = 2
     */
    tail?: StreamPosition;
}
declare class StreamPosition$Type extends MessageType<StreamPosition> {
    constructor();
    create(value?: PartialMessage<StreamPosition>): StreamPosition;
    internalBinaryRead(reader: IBinaryReader, length: number, options: BinaryReadOptions, target?: StreamPosition): StreamPosition;
    internalBinaryWrite(message: StreamPosition, writer: IBinaryWriter, options: BinaryWriteOptions): IBinaryWriter;
}
/**
 * @generated MessageType for protobuf message s2.v1.StreamPosition
 */
export declare const StreamPosition: StreamPosition$Type;
declare class Header$Type extends MessageType<Header> {
    constructor();
    create(value?: PartialMessage<Header>): Header;
    internalBinaryRead(reader: IBinaryReader, length: number, options: BinaryReadOptions, target?: Header): Header;
    internalBinaryWrite(message: Header, writer: IBinaryWriter, options: BinaryWriteOptions): IBinaryWriter;
}
/**
 * @generated MessageType for protobuf message s2.v1.Header
 */
export declare const Header: Header$Type;
declare class AppendRecord$Type extends MessageType<AppendRecord> {
    constructor();
    create(value?: PartialMessage<AppendRecord>): AppendRecord;
    internalBinaryRead(reader: IBinaryReader, length: number, options: BinaryReadOptions, target?: AppendRecord): AppendRecord;
    internalBinaryWrite(message: AppendRecord, writer: IBinaryWriter, options: BinaryWriteOptions): IBinaryWriter;
}
/**
 * @generated MessageType for protobuf message s2.v1.AppendRecord
 */
export declare const AppendRecord: AppendRecord$Type;
declare class AppendInput$Type extends MessageType<AppendInput> {
    constructor();
    create(value?: PartialMessage<AppendInput>): AppendInput;
    internalBinaryRead(reader: IBinaryReader, length: number, options: BinaryReadOptions, target?: AppendInput): AppendInput;
    internalBinaryWrite(message: AppendInput, writer: IBinaryWriter, options: BinaryWriteOptions): IBinaryWriter;
}
/**
 * @generated MessageType for protobuf message s2.v1.AppendInput
 */
export declare const AppendInput: AppendInput$Type;
declare class AppendAck$Type extends MessageType<AppendAck> {
    constructor();
    create(value?: PartialMessage<AppendAck>): AppendAck;
    internalBinaryRead(reader: IBinaryReader, length: number, options: BinaryReadOptions, target?: AppendAck): AppendAck;
    internalBinaryWrite(message: AppendAck, writer: IBinaryWriter, options: BinaryWriteOptions): IBinaryWriter;
}
/**
 * @generated MessageType for protobuf message s2.v1.AppendAck
 */
export declare const AppendAck: AppendAck$Type;
declare class SequencedRecord$Type extends MessageType<SequencedRecord> {
    constructor();
    create(value?: PartialMessage<SequencedRecord>): SequencedRecord;
    internalBinaryRead(reader: IBinaryReader, length: number, options: BinaryReadOptions, target?: SequencedRecord): SequencedRecord;
    internalBinaryWrite(message: SequencedRecord, writer: IBinaryWriter, options: BinaryWriteOptions): IBinaryWriter;
}
/**
 * @generated MessageType for protobuf message s2.v1.SequencedRecord
 */
export declare const SequencedRecord: SequencedRecord$Type;
declare class ReadBatch$Type extends MessageType<ReadBatch> {
    constructor();
    create(value?: PartialMessage<ReadBatch>): ReadBatch;
    internalBinaryRead(reader: IBinaryReader, length: number, options: BinaryReadOptions, target?: ReadBatch): ReadBatch;
    internalBinaryWrite(message: ReadBatch, writer: IBinaryWriter, options: BinaryWriteOptions): IBinaryWriter;
}
/**
 * @generated MessageType for protobuf message s2.v1.ReadBatch
 */
export declare const ReadBatch: ReadBatch$Type;
export {};
//# sourceMappingURL=s2.d.ts.map