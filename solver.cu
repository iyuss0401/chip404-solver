// Chip404 free-lane PoW solver (CUDA)
// h = keccak256( salt32 || sender20 || nonce32BE ), salt = keccak256("chip404.trial.v1")
// pass when uint256(h) <= threshold. Dual threshold: plain (target) + margin (target>>M).
// nonce layout: bits 71..64 = shard byte, bits 63..0 = counter.
// Full nonce = (shard << 64) | counter -- the driver reconstructs and verifies it.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <unistd.h>

typedef unsigned long long u64;
typedef unsigned int u32;

__device__ __forceinline__ u64 rotl64(u64 x, int y){ return (x<<y) | (x>>(64-y)); }

__device__ void keccakf(u64 st[25]){
    const u64 RC[24] = {
        0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
        0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
        0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
        0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
        0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
        0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL};
    const int rotc[24] = {1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44};
    const int piln[24] = {10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1};
    u64 bc[5];
    for(int round=0; round<24; round++){
        for(int i=0;i<5;i++) bc[i]=st[i]^st[i+5]^st[i+10]^st[i+15]^st[i+20];
        for(int i=0;i<5;i++){ u64 t=bc[(i+4)%5]^rotl64(bc[(i+1)%5],1); for(int j=0;j<25;j+=5) st[j+i]^=t; }
        u64 t=st[1];
        for(int i=0;i<24;i++){ int j=piln[i]; bc[0]=st[j]; st[j]=rotl64(t,rotc[i]); t=bc[0]; }
        for(int j=0;j<25;j+=5){ for(int i=0;i<5;i++) bc[i]=st[j+i]; for(int i=0;i<5;i++) st[j+i]^= (~bc[(i+1)%5]) & bc[(i+2)%5]; }
        st[0]^=RC[round];
    }
}

// constant lanes 0..6 (bytes 0..55 of the 84-byte input, nonce bytes 0..3 are zero)
__constant__ u64 c_lanes[7];
__constant__ uchar c_thrPlain[32];
__constant__ uchar c_thrMargin[32];

__device__ __forceinline__ int hashLeq(const u64 st[25], const uchar thr[32]){
    // compare squeezed hash bytes (BE integer) against threshold bytes
    for(int i=0;i<32;i++){
        uchar hb = (uchar)(st[i>>3] >> (8*(i&7)));
        if(hb < thr[i]) return 1;
        if(hb > thr[i]) return 0;
    }
    return 1; // equal passes (<=)
}

extern "C" __global__ void searchKernel(u64 startCounter, u64 itersPerThread, u32 shard,
                                        u64* foundMargin, u64* foundPlain,
                                        u64* bestKey, u64* bestCounter){
    u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 grid = (u64)gridDim.x * blockDim.x;
    u64 counter = startCounter + tid;
    for(u64 it=0; it<itersPerThread; it++, counter += grid){
        if(foundMargin[0]) return;
        u64 st[25];
        #pragma unroll
        for(int i=0;i<25;i++) st[i]=0;
        #pragma unroll
        for(int i=0;i<7;i++) st[i]=c_lanes[i];
        // lane 9 (message bytes 72..79): shard at byte 75 (lane byte 3);
        // counter bits 63..32 (hi32) as big-endian bytes 76..79 -> byte-swapped into lane bytes 4..7
        u32 hi = (u32)(counter >> 32);
        u64 l9hi = (u64)__byte_perm(hi, 0, 0x0123);
        st[9] = ((u64)shard << 24) | (l9hi << 32);
        // lane 10 (bytes 80..83): counter low 32 bits big-endian -> byte-swap; padding 0x01 at byte 84
        u32 lo = (u32)(counter & 0xFFFFFFFFULL);
        u64 l10 = (u64)__byte_perm(lo, 0, 0x0123);
        st[10] = l10 | (1ULL<<32);
        // lane 16: final padding bit at byte 135
        st[16] = 0x8000000000000000ULL;
        keccakf(st);
        // best tracking on top 64 bits of hash (BE)
        u64 key = 0;
        for(int i=0;i<8;i++){ uchar hb=(uchar)(st[i>>3]>>(8*(i&7))); key = (key<<8)|hb; }
        u64 old = atomicMin(bestKey, key);
        if(key < old){ bestCounter[0] = counter; }
        if(hashLeq(st, c_thrMargin)){
            if(atomicExch(foundMargin, 1ULL)==0ULL) foundMargin[1]=counter;
            return;
        }
        if(!foundPlain[0] && hashLeq(st, c_thrPlain)){
            if(atomicExch(foundPlain, 1ULL)==0ULL) foundPlain[1]=counter;
        }
    }
}

static void hexToBytes(const char* hex, uchar* out, int n){
    for(int i=0;i<n;i++){ unsigned int v; sscanf(hex+2*i, "%2x", &v); out[i]=(uchar)v; }
}
static int leadingZeroBitsKey(u64 key){ if(!key) return 64; int n=0; for(int i=63;i>=0;i--){ if(key>>i & 1ULL) break; n++; } return n; }
static void writeHitFile(const char* outPrefix, const char* kind, u64 counter, u32 shard){
    char path[512]; snprintf(path,sizeof(path),"%s_%s.txt",outPrefix,kind);
    FILE* f=fopen(path,"w");
    if(f){ fprintf(f,"counter=%llu\nshard=%u\n",(unsigned long long)counter,shard); fclose(f); }
}

int main(int argc, char** argv){
    // args: saltHex(64) senderHex(40) thrPlainHex(64) thrMarginHex(64) shard maxSeconds outPrefix [graceSeconds]
    if(argc < 8){ fprintf(stderr, "usage: solver salt sender thrPlain thrMargin shard maxSeconds outPrefix [graceSeconds]\n"); return 64; }
    uchar salt[32], sender[20], thrP[32], thrM[32];
    hexToBytes(argv[1], salt, 32);
    hexToBytes(argv[2], sender, 20);
    hexToBytes(argv[3], thrP, 32);
    hexToBytes(argv[4], thrM, 32);
    u32 shard = (u32)atoi(argv[5]);
    long maxSeconds = atol(argv[6]);
    const char* outPrefix = argv[7];
    long graceSeconds = (argc >= 9) ? atol(argv[8]) : 1200;

    // build input bytes 0..55 -> lanes 0..6
    uchar head[56]; memset(head,0,sizeof(head));
    memcpy(head, salt, 32); memcpy(head+32, sender, 20);
    u64 lanes[7];
    for(int i=0;i<7;i++){ u64 v=0; for(int b=0;b<8;b++) v |= ((u64)head[i*8+b])<<(8*b); lanes[i]=v; }
    cudaMemcpyToSymbol(c_lanes, lanes, sizeof(lanes));
    cudaMemcpyToSymbol(c_thrPlain, thrP, 32);
    cudaMemcpyToSymbol(c_thrMargin, thrM, 32);

    u64 *dMargin,*dPlain,*dBestKey,*dBestCounter;
    cudaMalloc(&dMargin, 16); cudaMalloc(&dPlain, 16); cudaMalloc(&dBestKey, 8); cudaMalloc(&dBestCounter, 8);
    u64 zero2[2]={0,0}; u64 big=~0ULL;
    cudaMemcpy(dMargin, zero2, 16, cudaMemcpyHostToDevice);
    cudaMemcpy(dPlain, zero2, 16, cudaMemcpyHostToDevice);
    cudaMemcpy(dBestKey, &big, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dBestCounter, zero2, 8, cudaMemcpyHostToDevice);

    const int BLOCK=256, GRID=1024;
    const u64 ITERS=1024; // per launch: GRID*BLOCK*ITERS = 2^28 hashes
    u64 startCounter=0, total=0;
    time_t t0=time(NULL);
    time_t plainAt=0;
    double rate=0;
    while(true){
        searchKernel<<<GRID, BLOCK>>>(startCounter, ITERS, shard, dMargin, dPlain, dBestKey, dBestCounter);
        cudaError_t err = cudaDeviceSynchronize();
        if(err != cudaSuccess){ fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); return 70; }
        total += (u64)GRID*BLOCK*ITERS;
        startCounter += (u64)GRID*BLOCK*ITERS;
        double el = difftime(time(NULL), t0); if(el>0) rate = total/el;
        u64 hMargin[2], hPlain[2], hBestKey, hBestCounter;
        cudaMemcpy(hMargin, dMargin, 16, cudaMemcpyDeviceToHost);
        cudaMemcpy(hPlain, dPlain, 16, cudaMemcpyDeviceToHost);
        cudaMemcpy(&hBestKey, dBestKey, 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(&hBestCounter, dBestCounter, 8, cudaMemcpyDeviceToHost);
        printf("[shard %u] checked %.3f G hashes, %.3f GH/s, best top zero bits >= %d (counter %llu)\n",
               shard, total/1e9, rate/1e9, leadingZeroBitsKey(hBestKey), (unsigned long long)hBestCounter);
        fflush(stdout);
        if(hMargin[0]){
            writeHitFile(outPrefix, "margin", hMargin[1], shard);
            printf("FOUND_MARGIN counter=%llu shard=%u\n",(unsigned long long)hMargin[1],shard); fflush(stdout);
            return 0;
        }
        if(hPlain[0] && !plainAt){
            plainAt = time(NULL);
            writeHitFile(outPrefix, "plain", hPlain[1], shard); // visible to the driver immediately
            printf("FOUND_PLAIN counter=%llu shard=%u (margin hunt continues up to %ld s grace)\n",
                   (unsigned long long)hPlain[1], shard, graceSeconds); fflush(stdout);
        }
        long deadline = maxSeconds;
        if(plainAt){
            long untilGraceEnd = (long)((plainAt - t0) + graceSeconds);
            if(untilGraceEnd < deadline) deadline = untilGraceEnd;
        }
        if(el >= deadline){
            if(hPlain[0]){
                printf("STOP plain only counter=%llu shard=%u\n",(unsigned long long)hPlain[1],shard); fflush(stdout);
                return 2;
            }
            printf("TIMEOUT no hit. best counter=%llu\n",(unsigned long long)hBestCounter); fflush(stdout);
            return 3;
        }
    }
}
