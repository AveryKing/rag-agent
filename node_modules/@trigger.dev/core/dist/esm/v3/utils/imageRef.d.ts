export type DockerImageParts = {
    registry?: string;
    repo: string;
    tag?: string;
    digest?: string;
};
export declare function parseDockerImageReference(imageReference: string): DockerImageParts;
export declare function rebuildDockerImageReference(parts: DockerImageParts): string;
