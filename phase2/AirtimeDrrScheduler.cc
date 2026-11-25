#include "AirtimeDrrScheduler.h"

namespace ca2 {

Define_Module(AirtimeDrrScheduler);

void AirtimeDrrScheduler::initialize(int stage)
{
    AirtimeSchedulerBase::initialize(stage);
    if (stage == INITSTAGE_LOCAL) {
        quantum = par("quantum");
        deficit.assign(getNumQueues(), SIMTIME_ZERO);
        WATCH_VECTOR(deficit);
    }
}

// Called once each time the MAC takes a frame. Return the index of the queue
// to serve, or -1 if every queue is empty. Available helpers (see
// AirtimeSchedulerBase.h): getNumQueues(), hasPacket(i), headAirtime(i).
int AirtimeDrrScheduler::schedulePacket()
{
    int n = getNumQueues();
    bool anyPacket = false;
    for (int i = 0; i < n; i++) {
        if (hasPacket(i)) {
            anyPacket = true;
            break;
        }
    }
    if (!anyPacket)
        return -1;

    while (true) {
        if (!hasPacket(current)) {
            deficit[current] = SIMTIME_ZERO;
            current = (current + 1) % n;
        }
        else if (deficit[current] <= SIMTIME_ZERO) {
            deficit[current] += quantum;
            current = (current + 1) % n;
        }
        else {
            deficit[current] -= headAirtime(current);
            return current;
        }
    }
}

} // namespace ca2
