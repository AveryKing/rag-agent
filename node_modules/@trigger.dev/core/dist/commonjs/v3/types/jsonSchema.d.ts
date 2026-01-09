/**
 * JSON Schema type definition - compatible with JSON Schema Draft 7
 * Based on the JSONSchema7 type from @types/json-schema but defined inline to avoid import issues
 */
export interface JSONSchema {
    $id?: string;
    $ref?: string;
    $schema?: string;
    $comment?: string;
    type?: JSONSchemaType | JSONSchemaType[];
    enum?: any[];
    const?: any;
    multipleOf?: number;
    maximum?: number;
    exclusiveMaximum?: number;
    minimum?: number;
    exclusiveMinimum?: number;
    maxLength?: number;
    minLength?: number;
    pattern?: string;
    format?: string;
    items?: JSONSchema | JSONSchema[];
    additionalItems?: JSONSchema | boolean;
    maxItems?: number;
    minItems?: number;
    uniqueItems?: boolean;
    contains?: JSONSchema;
    maxProperties?: number;
    minProperties?: number;
    required?: string[];
    properties?: Record<string, JSONSchema>;
    patternProperties?: Record<string, JSONSchema>;
    additionalProperties?: JSONSchema | boolean;
    dependencies?: Record<string, JSONSchema | string[]>;
    propertyNames?: JSONSchema;
    if?: JSONSchema;
    then?: JSONSchema;
    else?: JSONSchema;
    allOf?: JSONSchema[];
    anyOf?: JSONSchema[];
    oneOf?: JSONSchema[];
    not?: JSONSchema;
    title?: string;
    description?: string;
    default?: any;
    readOnly?: boolean;
    writeOnly?: boolean;
    examples?: any[];
    [key: string]: any;
}
export type JSONSchemaType = "string" | "number" | "integer" | "boolean" | "object" | "array" | "null";
