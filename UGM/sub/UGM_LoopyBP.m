
<!-- saved from url=(0065)http://www.di.ens.fr/~mschmidt/Software/UGM/updates/UGM_LoopyBP.m -->
<html><head><meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1"></head><body><pre style="word-wrap: break-word; white-space: pre-wrap;">function [new_msg] = UGM_LoopyBP(nodePot,edgePot,edgeStruct,maximize)

[nNodes,maxState] = size(nodePot);
nEdges = size(edgePot,3);
edgeEnds = edgeStruct.edgeEnds;
V = edgeStruct.V;
E = edgeStruct.E;
nStates = double(edgeStruct.nStates);

% Initialize
nodeBel = zeros(nNodes,maxState);
oldNodeBel = nodeBel;
prod_of_msgs = zeros(maxState,nNodes);
old_msg = zeros(maxState,nEdges*2);
new_msg = zeros(maxState,nEdges*2);
for e = 1:nEdges
    n1 = edgeEnds(e,1);
    n2 = edgeEnds(e,2);
    new_msg(1:nStates(n2),e) = 1/nStates(n2); % Message from n1 =&gt; n2
    new_msg(1:nStates(n1),e+nEdges) = 1/nStates(n1); % Message from n2 =&gt; n1
end

for i = 1:edgeStruct.maxIter
    for n = 1:nNodes
        % Find Neighbors
        edges = UGM_getEdges(n,edgeStruct);

        % Send a message to each neighbor
        for e = edges
            n1 = edgeEnds(e,1);
            n2 = edgeEnds(e,2);

            if n == edgeEnds(e,2)
                pot_ij = edgePot(1:nStates(n1),1:nStates(n2),e);
            else
                pot_ij = edgePot(1:nStates(n1),1:nStates(n2),e)';
            end

            % Compute temp = product of all incoming msgs except j
            temp = nodePot(n,1:nStates(n))';
            for e2 = edges
                if e ~= e2
                    if n == edgeEnds(e2,2)
                        temp = temp .* new_msg(1:nStates(n),e2);
                    else
                        temp = temp .* new_msg(1:nStates(n),e2+nEdges);
                    end
                end
            end

            % Compute new message
            if maximize
				if edgeStruct.useMex
					newm = max_mult(pot_ij,temp);
				else
					newm = max_multM(pot_ij,temp);
				end
            else
                newm = pot_ij * temp;
			end

            if n == edgeEnds(e,2);
                new_msg(1:nStates(n1),e+nEdges) = newm./sum(newm);
            else
                new_msg(1:nStates(n2),e) = newm./sum(newm);
            end
        end
    end

    % Check convergence
    if sum(abs(new_msg(:)-old_msg(:))) &lt; 1e-4
        break;
    end

    old_msg = new_msg;
end
if i == edgeStruct.maxIter
    fprintf('Loopy did not converge\n');
end
</pre></body></html>